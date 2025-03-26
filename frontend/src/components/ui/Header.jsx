import React from "react";
import { Link } from "react-router-dom";
import useAuthStore from "../context/authStore.js";
// import Logo from '../../assets/images/logo.svg';

export default function Header() {
  const { user, logout } = useAuthStore();
 
  return (
    <header className="border-b border-[#bbbbbb] w-full">
      <div className="w-full pl-0 pr-4 py-4 flex justify-between items-center">
        <Link to="/" className="flex items-center">
          <img
            src="/Fundit_logo.png" 
            alt="Fundit 로고" 
            style={{ height: '25px', width: 'auto', cursor: 'pointer', marginLeft: '55%' }} 
          />
        </Link>
        <nav className="space-x-6 text-[#8a8a8a]" style={{ marginRight: '3%' }}>
          {user ? (
            // 로그인 상태
            <>
              <span>{user.username}님, 환영합니다!</span>
              <Link to="/mypage" className="hover:text-[#4ba6f7]">마이페이지</Link>
              <Link to="/calendar" className="hover:text-[#4ba6f7]">캘린더</Link>
              <button onClick={logout} className="hover:text-[#4ba6f7]">로그아웃</button>
              <Link to="/services" className="hover:text-[#4ba6f7]">회사 소개</Link>
            </>
          ) : (
            // 로그아웃 상태
            <>
             <Link to="/login" className="hover:text-[#4ba6f7]">로그인 </Link>
             <Link to="/signup" className="hover:text-[#4ba6f7]">회원가입 </Link>
             <Link to="/services" className="hover:text-[#4ba6f7]">회사 소개</Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
