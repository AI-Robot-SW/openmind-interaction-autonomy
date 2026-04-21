export default function StandbyPanel() {
  return (
    <div style={{
      position: 'absolute',
      top: '0',
      left: '0',
      width: '100vw',
      height: '100vh',
      zIndex: 25,
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: 'rgba(0, 0, 0, 1)'
    }}>
      {/* 크게 확대된 로딩 인디케이터 */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        gap: '48px' // 기존 8px에서 24px로 확대
      }}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            width: '160px',        
            height: '160px',      
            borderRadius: '50%',
            backgroundColor: '#FFD700',
            animation: `bounce 1.5s infinite ${i * 0.2}s`
          }} />
        ))}
      </div>
      
      <style>
        {`
          @keyframes bounce {
            0%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-20px); }
          }
        `}
      </style>
    </div>
  );
};

