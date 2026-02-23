"""
Unitree Go2 Provider - Go2 control interface (SDK direct, no ROS).

Wraps Unitree Python SDK (unitree_sdk2py) SportClient to expose sport mode APIs
such as Move, StopMove, StandUp, Damp, Sit.

Also subscribes to SportModeState for odometry data (position, yaw, velocity).
"""

import logging
import threading
import time
from typing import Optional
from dataclasses import dataclass

try:
    from unitree.unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
except ImportError:
    ChannelFactoryInitialize = None  # type: ignore
    ChannelSubscriber = None  # type: ignore

try:
    from unitree.unitree_sdk2py.go2.sport.sport_client import SportClient
except ImportError:
    SportClient = None  # type: ignore

try:
    from unitree.unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
except ImportError:
    SportModeState_ = None  # type: ignore

from .singleton import singleton


@dataclass(frozen=True)
class OdometryData:
    """
    Odometry data from Go2 robot.
    
    Attributes
    ----------
    t_monotonic : Optional[float]
        Monotonic timestamp (time.monotonic()) when data was received
    x : Optional[float]
        X position in meters
    y : Optional[float]
        Y position in meters
    yaw : Optional[float]
        Yaw angle in radians
    vx : Optional[float]
        Linear velocity in x direction (m/s)
    vy : Optional[float]
        Linear velocity in y direction (m/s)
    vyaw : Optional[float]
        Angular velocity around z axis (rad/s)
    """
    t_monotonic: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    yaw: Optional[float] = None
    vx: Optional[float] = None
    vy: Optional[float] = None
    vyaw: Optional[float] = None


@singleton
class UnitreeGo2Provider:
    """
    Unitree Go2 control Provider.

    Wraps Go2 sport mode API (SDK). Singleton: a single control channel is
    shared across the system.

    Parameters
    ----------
    channel : str
        CycloneDDS/Unitree Ethernet channel (e.g. "eth0"). If empty, channel
        init is skipped.
    timeout : float
        RPC timeout in seconds for SportClient (e.g. 10.0). Used for SetTimeout.
    state_topic : str
        DDS topic name for SportModeState data (default: "rt/sportmodestate")
    """

    def __init__(
        self, 
        channel: str = "", 
        timeout: float = 10.0,
        state_topic: str = "rt/sportmodestate"
    ) -> None:
        """
        Initialize the Unitree Go2 Provider.

        Parameters
        ----------
        channel : str
            CycloneDDS/Unitree Ethernet channel (e.g. "eth0"). If empty, channel init is skipped.
        timeout : float
            RPC timeout in seconds for SportClient (default 10.0).
        state_topic : str
            DDS topic name for SportModeState (default: "rt/sportmodestate")
        """
        self._channel = channel
        self._timeout = timeout
        self._state_topic = state_topic
        self._sport_client: Optional[SportClient] = None
        self._state_subscriber: Optional[ChannelSubscriber] = None
        self._lock = threading.Lock()
        self._running: bool = False
        self._data: Optional[dict] = None
        
        # Odometry data storage
        self._odometry_lock = threading.Lock()
        self._odometry_data: Optional[OdometryData] = None

        logging.info("UnitreeGo2Provider initialized (channel=%r, timeout=%s)", channel, timeout)

    def _sportmode_callback(self, msg: "SportModeState_") -> None:
        """
        Callback for SportModeState messages.
        
        This is called by ChannelSubscriber whenever new state data arrives.
        Frequency depends on the publisher (typically 10-30 Hz).
        
        Parameters
        ----------
        msg : SportModeState_
            SportModeState message from DDS
        """
        try:
            # Get monotonic timestamp
            t_monotonic = time.monotonic()
            
            # Extract position (x, y) - meters
            x = float(msg.position[0]) if len(msg.position) > 0 else 0.0
            y = float(msg.position[1]) if len(msg.position) > 1 else 0.0
            
            # Extract yaw from IMU (radians)
            yaw = float(msg.imu_state.rpy[2]) if len(msg.imu_state.rpy) > 2 else 0.0
            
            # Extract velocities (m/s)
            vx = float(msg.velocity[0]) if hasattr(msg, 'velocity') and len(msg.velocity) > 0 else 0.0
            vy = float(msg.velocity[1]) if hasattr(msg, 'velocity') and len(msg.velocity) > 1 else 0.0
            
            # Extract angular velocity from gyroscope (rad/s)
            vyaw = float(msg.imu_state.gyroscope[2]) if len(msg.imu_state.gyroscope) > 2 else 0.0
            
            # Create new frozen dataclass instance
            odom_data = OdometryData(
                t_monotonic=t_monotonic,
                x=x,
                y=y,
                yaw=yaw,
                vx=vx,
                vy=vy,
                vyaw=vyaw
            )
            
            # Update odometry data (thread-safe)
            with self._odometry_lock:
                self._odometry_data = odom_data
                
        except Exception as e:
            logging.error("UnitreeGo2Provider: SportModeState callback error: %s", e)

    def start(self) -> None:
        """
        Start the provider.

        Initializes SportClient and subscribes to SportModeState topic.
        """
        with self._lock:
            if self._running:
                return

            if SportClient is None:
                logging.error(
                    "UnitreeGo2Provider: unitree_sdk2py not available. Install unitree_sdk2py."
                )
                return

            if self._channel and ChannelFactoryInitialize is not None:
                try:
                    ChannelFactoryInitialize(0, self._channel)  # type: ignore
                except Exception as e:
                    logging.error(
                        "UnitreeGo2Provider: ChannelFactoryInitialize failed: %s", e
                    )
                    return

            try:
                self._sport_client = SportClient()
                self._sport_client.SetTimeout(self._timeout)
                self._sport_client.Init()
                self._sport_client.StopMove()
                logging.info("UnitreeGo2Provider: SportClient initialized")
            except Exception as e:
                logging.error(f"UnitreeGo2Provider: Failed to init SportClient: {e}")
                self._sport_client = None
                return

            # Subscribe to SportModeState topic
            if ChannelSubscriber is not None and SportModeState_ is not None:
                try:
                    self._state_subscriber = ChannelSubscriber(
                        self._state_topic, 
                        SportModeState_
                    )
                    # Init with callback and queue size of 10
                    self._state_subscriber.Init(self._sportmode_callback, 10)
                    logging.info(
                        "UnitreeGo2Provider: SportModeState subscriber initialized (topic=%s)", 
                        self._state_topic
                    )
                except Exception as e:
                    logging.error(
                        "UnitreeGo2Provider: Failed to init SportModeState subscriber: %s", e
                    )
                    self._state_subscriber = None
            else:
                logging.warning(
                    "UnitreeGo2Provider: SportModeState subscriber not available "
                    "(ChannelSubscriber or SportModeState_ not imported)"
                )

            self._running = True
            self._data = {"initialized": True}
            logging.info("UnitreeGo2Provider started")

    def stop(self) -> bool:
        """
        Stop the provider.

        Calls StopMove and clears state. Does not destroy SportClient (SDK may not
        support re-init). Returns True if StopMove succeeded or no client was set,
        False on RPC error or exception.

        Returns
        -------
        bool
            True if stop completed (StopMove accepted or nothing to send), False otherwise.
        """
        with self._lock:
            self._running = False
        self._data = None
        
        # Note: ChannelSubscriber doesn't need explicit cleanup in SDK2
        # The subscriber will stop receiving when the object is garbage collected
        
        if self._sport_client is None:
            logging.info("UnitreeGo2Provider stopped")
            return True
        try:
            code = self._sport_client.StopMove()
            if code != 0:
                logging.error("UnitreeGo2Provider: StopMove on stop failed (code=%s)", code)
                logging.info("UnitreeGo2Provider stopped")
                return False
            logging.info("UnitreeGo2Provider stopped")
            return True
        except Exception as e:
            logging.error("UnitreeGo2Provider: StopMove on stop: %s", e)
            logging.info("UnitreeGo2Provider stopped")
            return False

    def get_odometry(self) -> Optional[OdometryData]:
        """
        Get latest odometry data.
        
        Returns the current odometry data from SportModeState. The data is updated
        asynchronously by the DDS subscriber (typically at 10-30 Hz).
        Since OdometryData is frozen, the returned reference is safe to use.
        
        Robot behavior: None. Read-only access to odometry data received
        from the robot's SportModeState topic.
        
        Returns
        -------
        Optional[OdometryData]
            Current odometry data (x, y, yaw, velocities), or None if no data received yet.
        """
        with self._odometry_lock:
            return self._odometry_data

    def move(self, vx: float, vy: float, vyaw: float) -> bool:
        """
        Send velocity command (body frame). Last command is held until stop_move.

        Robot behavior: Drives the robot at the given forward (vx), lateral (vy),
        and yaw (vyaw) velocities in body frame. The robot keeps this motion until
        stop_move or a new move is sent. NoReply: no acknowledgment from the robot.

        The SDK Move is NoReply; the return value from the robot is not checked.
        Returns True if the send completed without exception, False otherwise.

        Parameters
        ----------
        vx : float
            Forward velocity (m/s).
        vy : float
            Lateral velocity (m/s).
        vyaw : float
            Yaw rate (rad/s).

        Returns
        -------
        bool
            True if the command was sent without exception, False otherwise.
        """
        if self._sport_client is None:
            logging.warning("UnitreeGo2Provider: move ignored (SportClient not ready)")
            return False
        try:
            self._sport_client.Move(vx, vy, vyaw)
            return True
        except Exception as e:
            logging.error("UnitreeGo2Provider: Move failed: %s", e)
            return False

    def stop_move(self) -> bool:
        """
        Stop movement (velocity command zero).

        Robot behavior: Stops all locomotion. Sets velocity to zero so the robot
        stops walking/running in place. Does not change posture (stand/sit/damp).

        Returns
        -------
        bool
            True if the command was accepted, False otherwise.
        """
        if self._sport_client is None:
            return False
        try:
            code = self._sport_client.StopMove()
            if code != 0:
                logging.error("UnitreeGo2Provider: StopMove failed (code=%s)", code)
                return False
            return True
        except Exception as e:
            logging.error("UnitreeGo2Provider: StopMove failed: %s", e)
            return False

    def stand_up(self) -> bool:
        """
        Stand up from crouch/sit.

        Robot behavior: Makes the robot rise from a lying or crouching posture
        to a standing posture. Use after stand_down or damp (e.g. after carrying).

        Returns
        -------
        bool
            True if the command was accepted, False otherwise.
        """
        if self._sport_client is None:
            logging.warning("UnitreeGo2Provider: stand_up ignored (SportClient not ready)")
            return False
        try:
            code = self._sport_client.StandUp()
            if code != 0:
                logging.error("UnitreeGo2Provider: StandUp failed (code=%s)", code)
                return False
            return True
        except Exception as e:
            logging.error("UnitreeGo2Provider: StandUp failed: %s", e)
            return False

    def stand_down(self) -> bool:
        """
        Stand down (crouch/lie).

        Robot behavior: Lowers the robot from standing to a crouching or lying
        posture. Joints are still under control (not fully passive).

        Returns
        -------
        bool
            True if the command was accepted, False otherwise.
        """
        if self._sport_client is None:
            return False
        try:
            code = self._sport_client.StandDown()
            if code != 0:
                logging.error("UnitreeGo2Provider: StandDown failed (code=%s)", code)
                return False
            return True
        except Exception as e:
            logging.error("UnitreeGo2Provider: StandDown failed: %s", e)
            return False

    def damp(self) -> bool:
        """
        Enter damping mode (e.g. for carrying).

        Robot behavior: Puts joints into a passive/damping mode so the robot
        can be moved by hand (e.g. carried). Motors resist little; use stand_up
        to return to normal control.

        Returns
        -------
        bool
            True if the command was accepted, False otherwise.
        """
        if self._sport_client is None:
            return False
        try:
            code = self._sport_client.Damp()
            if code != 0:
                logging.error("UnitreeGo2Provider: Damp failed (code=%s)", code)
                return False
            return True
        except Exception as e:
            logging.error("UnitreeGo2Provider: Damp failed: %s", e)
            return False

    def sit(self) -> bool:
        """
        Sit down.

        Robot behavior: Makes the robot sit on the ground (legs folded). Use
        rise_sit to stand again from sit.

        Returns
        -------
        bool
            True if the command was accepted, False otherwise.
        """
        if self._sport_client is None:
            return False
        try:
            code = self._sport_client.Sit()
            if code != 0:
                logging.error("UnitreeGo2Provider: Sit failed (code=%s)", code)
                return False
            return True
        except Exception as e:
            logging.error("UnitreeGo2Provider: Sit failed: %s", e)
            return False

    def rise_sit(self) -> bool:
        """
        Rise from sit to stand.

        Robot behavior: Makes the robot stand up from a sitting posture. Use
        after sit() when you want the robot back on its feet.

        Returns
        -------
        bool
            True if the command was accepted, False otherwise.
        """
        if self._sport_client is None:
            return False
        try:
            code = self._sport_client.RiseSit()
            if code != 0:
                logging.error("UnitreeGo2Provider: RiseSit failed (code=%s)", code)
                return False
            return True
        except Exception as e:
            logging.error("UnitreeGo2Provider: RiseSit failed: %s", e)
            return False

    def recovery_stand(self) -> bool:
        """
        Recovery stand.

        Robot behavior: Recovery/self-righting motion. Use when the robot has
        fallen or is in an abnormal posture to attempt standing again.

        Returns
        -------
        bool
            True if the command was accepted, False otherwise.
        """
        if self._sport_client is None:
            return False
        try:
            code = self._sport_client.RecoveryStand()
            if code != 0:
                logging.error("UnitreeGo2Provider: RecoveryStand failed (code=%s)", code)
                return False
            return True
        except Exception as e:
            logging.error("UnitreeGo2Provider: RecoveryStand failed: %s", e)
            return False

    def heartbeat(self) -> bool:
        """
        Check RPC connectivity with the robot (heartbeat).

        Robot behavior: None. This only queries the robot (read-only AutoRecoveryGet).
        Use it periodically to verify that the OM process can still get a response
        from the robot within the configured timeout.

        Returns
        -------
        bool
            True if the robot responded (code == 0), False otherwise.
        """
        if self._sport_client is None:
            return False
        try:
            code, _ = self._sport_client.AutoRecoveryGet()
            return code == 0
        except Exception as e:
            logging.debug("UnitreeGo2Provider: heartbeat failed: %s", e)
            return False
    
    @property
    def data(self) -> Optional[dict]:
        """
        Get current provider data (e.g. initialized).

        Robot behavior: None. Read-only status from the provider (e.g. whether
        the provider has started and SportClient is initialized).

        Returns
        -------
        Optional[dict]
            Provider status, or None if not started.
        """
        return self._data