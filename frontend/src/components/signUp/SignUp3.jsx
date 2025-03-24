import React from 'react';
//import { Link } from 'react-router-dom';
import '../../output.css';

export default function SignUp3() {
  return (      
    <div className="min-c-screen bg-white">
      {/* Main Content */}
      <main className="flex flex-col items-center justify-center flex-1 px-6 mt-32">
        <h1 className="text-[#4ba6f7] text-4xl font-bold mb-4">가입완료</h1>
        <h2 className="text-[#00316b] text-3xl font-bold mb-12">전은수님 반가워요!</h2>
        <button className="bg-[#4ba6f7] text-white px-12 py-4 rounded-lg text-lg font-medium hover:bg-[#4ba6f7]/90 transition-colors">
          바로 채팅하기
        </button>
      </main>
    </div>
  );
}
