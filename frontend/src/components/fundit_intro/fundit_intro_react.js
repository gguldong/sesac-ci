import React, { useEffect } from 'react';
import './fundit_intro.css';
import calendarIcon from './Calendar icon.svg';
import userIcon from './User icon.svg';
import boltIcon from './Bolt icon.svg';
import introSvg from './intro.svg';

// 글로벌 스타일을 재정의하기 위한 컴포넌트
const GlobalStyles = () => {
  useEffect(() => {
    // 기존 스타일 저장
    const originalStyle = document.body.style.cssText;
    const htmlStyle = document.documentElement.style.cssText;
    const rootStyle = document.getElementById('root')?.style.cssText || '';
    
    // 스크롤 가능하도록 스타일 변경
    document.body.style.overflow = 'auto';
    document.documentElement.style.overflow = 'auto';
    if (document.getElementById('root')) {
      document.getElementById('root').style.overflow = 'auto';
      document.getElementById('root').style.height = 'auto';
      document.getElementById('root').style.minHeight = 'auto';
    }
    
    // 컴포넌트 언마운트 시 원래 스타일로 복원
    return () => {
      document.body.style.cssText = originalStyle;
      document.documentElement.style.cssText = htmlStyle;
      if (document.getElementById('root')) {
        document.getElementById('root').style.cssText = rootStyle;
      }
    };
  }, []);
  
  return null;
};

function FunditIntroReact() {
  useEffect(() => {
    // 선 애니메이션 효과
    const divider = document.querySelector('.intro-divider');
    if (divider) {
      // 초기 상태 설정
      divider.style.height = '0'; 
      divider.style.opacity = '0'; 
      divider.style.transition = 'none'; 

      // Intersection Observer 설정
      const observer = new IntersectionObserver(entries => {
          entries.forEach(entry => {
              if (entry.isIntersecting) {
                  // 선이 나타날 때 애니메이션 적용
                  divider.style.transition = 'height 1.2s ease, opacity 1.2s ease'; 
                  divider.style.height = '160px'; 
                  divider.style.opacity = '1'; 

                  observer.unobserve(divider); // 한 번만 실행되도록 관찰 중지
              }
          });
      });

      observer.observe(divider); // 선 관찰 시작
    }

    // info-boxes 애니메이션 효과
    const infoBoxes = document.querySelector('.intro-info-boxes');
    if (infoBoxes) {
      const infoBoxList = infoBoxes.querySelectorAll('.intro-info-box');

      const observer = new IntersectionObserver(entries => {
          entries.forEach(entry => {
              if (entry.isIntersecting) {
                  // info-boxes 요소가 화면에 나타났을 때 애니메이션 실행
                  infoBoxList.forEach((box, index) => {
                      setTimeout(() => {
                          box.style.opacity = '1';
                          box.style.transform = 'translateY(0)';
                      }, 500 * index); // 순차적인 지연 시간 조절
                  });
              } else {
                  // info-boxes 요소가 화면에서 사라졌을 때 초기 상태로 되돌림
                  infoBoxList.forEach(box => {
                      box.style.opacity = '0';
                      box.style.transform = 'translateY(50px)';
                  });
              }
          });
      });

      observer.observe(infoBoxes);
    }
  }, []);

  return (
    <>
      <GlobalStyles />
      <div className="intro-container" style={{ height: 'auto', overflow: 'auto', position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
        <div className="intro-content">
          <div className="intro-left-section">
            <h1>쉽고 빠르게 <span className="intro-highlight">정책</span>을 찾아 신청하고,<br />나만의 혜택을 누리세요.</h1>
            <div className="intro-divider"></div>
            <div className="intro-softtext">
              AI가 신청 가능 기간과 조건을 바로 알려줘요.<br />
              캘린더로 신청 기간을 한눈에 확인하고,<br />
              일일이 찾아볼 필요 없이 간편하게 혜택을 누릴 수 있습니다.
            </div>
            <div className="intro-icons">
              <div className="intro-icon">
                <img src={calendarIcon} alt="맞춤 추천 아이콘" />
                <p><span className="intro-bold-text">맞춤 추천</span></p>
                <p>나만의 맞춤 정책 추천</p>
              </div>
              <div className="intro-icon">
                <img src={userIcon} alt="신청 기간 안내 아이콘" />
                <p><span className="intro-bold-text">신청 기간 안내</span></p>
                <p>정책 신청 기간 캘린더로 한눈에 확인</p>
              </div>
              <div className="intro-icon">
                <img src={boltIcon} alt="빠른 정보 제공 아이콘" />
                <p><span className="intro-bold-text">빠른 정보 제공</span></p>
                <p>AI가 바로 제공해주는 정책 정보</p>
              </div>
            </div>
          </div>
          <div className="intro-right-section-wrapper">
            <div className="intro-right-section">
              <p className="intro-right-bold-text">일일이 찾아보지 않고<br />AI가 바로 알려주는 <span className="intro-highlight">맞춤 정책</span></p>
              <p className="intro-right-small-text">AI가 신청 가능 기간과 조건을 바로 알려줘요.<br />캘린더로 신청 기간을 한눈에 확인하고,<br />일일이 찾아볼 필요 없이 간편하게 혜택을 누릴 수 있습니다.</p>
              <img src={introSvg} alt="정책 알림 SVG" />
            </div>
          </div>
          <div className="intro-info">
            <p>믿을 수 있는 정책 정보, <span className="intro-highlight">Fundit과 함께하세요</span></p>
            <div className="intro-info-boxes">
              <div className="intro-info-box">"Fundit이 직접 검토한 정확한 정책 정보만 제공합니다."</div>
              <div className="intro-info-box">"사용자의 혜택을 최우선으로, 필요한 정책만 선별합니다."</div>
              <div className="intro-info-box">"정책 정보는 최신 데이터 기반으로 정확하게 제공됩니다."</div>
              <div className="intro-info-box">"Fundit과 함께라면 놓치는 정책 없이 혜택을 받을 수 있습니다."</div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default FunditIntroReact; 