import React from 'react';
import '../../output.css';
import '../ui.css';
import { Link } from 'react-router-dom';

export default function MainPage() {
  return (
    <div className="min-h-screen bg-[#f4f4f4] relative">
      <div className="main-image-container">
        <img 
          src="/mainimage.svg" 
          alt="펀딧 메인 이미지" 
          className="main-image"
        />
      </div>
      
      <main className="container mx-auto px-4 py-20">
        <div className="max-w-3xl">
          <h1 className="text-5xl font-bold leading-tight mb-8">
            <span className="text-[#4ba6f7]">펀딧</span>과 함께라면,
            <br />
            보조금 걱정 끝!
          </h1>

          <p className="text-[#8a8a8a] text-lg mb-4">
            펀딧은 다양한 보조금 정보를 한 곳에 모아 사용자에게 제공하고,
            <br />
            보조금 신청 과정을 간소화하여 빠르고 쉽게 지원을 받을 수 있도록
            <br />
            필요한 도움을 실시간으로 제공합니다.
          </p>

          <p className="text-[#8a8a8a] italic mb-8">
            "Pundit gathers subsidy information in one place,
            <br />
            streamlines the application process, and provides
            <br />
            real-time support for quick and easy access."
          </p>

          <Link to={"chat"} className="bg-[#4ba6f7] text-white px-8 py-4 rounded-lg text-lg font-medium hover:bg-[#00316b] transition-colors">
            바로 채팅하기
          </Link>
        </div>
      </main>
    </div>
  );
}
