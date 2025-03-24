import React from "react";
import { Link } from "react-router-dom";
import useAuthStore from "../../components/context/authStore.js";
// import Logo from '../../assets/images/logo.svg';

export default function Header() {
  const { user, logout } = useAuthStore();
 
  return (
    <header className="border-b border-[#bbbbbb]">
      <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
        <Link to="/" className="text-[#4ba6f7] text-3xl font-bold">
          Fundit
        </Link>
        <nav className="space-x-6 text-[#8a8a8a]">
          {user ? (
            // 로그인 상태
            <>
              <span>{user.username}님, 환영합니다!</span>
              <Link to="/mypage" className="hover:text-[#4ba6f7]">마이페이지</Link>
              <button onClick={logout} className="hover:text-[#4ba6f7]">로그아웃</button>
              <Link to="/services" className="hover:text-[#4ba6f7]">민원서비스</Link>
            </>
          ) : (
            // 로그아웃 상태
            <>
             <Link to="/login" className="hover:text-[#4ba6f7]">로그인 </Link>
             <Link to="/signup" className="hover:text-[#4ba6f7]">회원가입 </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
