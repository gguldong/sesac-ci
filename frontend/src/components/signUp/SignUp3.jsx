import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import useAuthStore from '../context/authStore'; // authStore를 가져옵니다.
import '../../output.css';

export default function SignUp3() {
  const navigate = useNavigate();
  const { user, fetchUser } = useAuthStore(); // authStore에서 user와 fetchUser를 가져옵니다.

  useEffect(() => {
    fetchUser(); // 컴포넌트가 마운트될 때 사용자 정보를 가져옵니다.
  }, [fetchUser]);

  return (      
    <div className="min-c-screen bg-white">
      {/* Main Content */}
      <main className="flex flex-col items-center justify-center flex-1 px-6 mt-32">
        <h1 className="text-[#4ba6f7] text-4xl font-bold mb-4">가입완료</h1>
        <h2 className="text-[#00316b] text-3xl font-bold mb-12">{user ? `${user.username}님 반가워요!` : '사용자 정보를 불러오는 중...'}</h2>
        <button onClick={() => navigate('chat')} className="bg-[#4ba6f7] text-white px-12 py-4 rounded-lg text-lg font-medium hover:bg-[#4ba6f7]/90 transition-colors">
          바로 채팅하기
        </button>
      </main>
    </div>
  );
}