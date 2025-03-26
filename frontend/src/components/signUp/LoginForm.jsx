import React, { useState } from "react";
//import { Checkbox } from "../ui/Checkbox";
import { Input } from "../ui/Input.jsx";
import { Link, useNavigate } from "react-router-dom";
import '../../output.css';
//import axios from "axios";
import useAuthStore from "../context/authStore.js";
// LoginForm component
export default function LoginForm() {
  const { login } = useAuthStore();
  const [user_id, setUser_id] = useState("");
  const [password, setPassword] = useState("");
  const [loginHovered, setLoginHovered] = useState(false);
  const [signupHovered, setSignupHovered] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (event) => {
    event.preventDefault()
    const success = await login(user_id, password);
    if (success) {
      alert("로그인 성공!");
      navigate("/");
    } else {
      alert("로그인 실패!");
    }
  };

  const buttonStyle = {
    width: '100%',
    height: '48px',
    backgroundColor: '#4ba6f7',
    color: 'white',
    borderRadius: '6px',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'all 0.2s ease-in-out',
  };

  const buttonHoverStyle = {
    backgroundColor: '#557499',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    transform: 'scale(1.02)',
  };
  
  return (
    <div className="min-h-screen bg-white">
      <main className="max-w-md mx-auto mt-20 px-4">
        <div className="text-center mb-16">
          <Link to="/">
            <img 
              src="/Fundit_logo.png" 
              alt="Fundit 로고" 
              className="w-auto h-20 mx-auto cursor-pointer" 
            />
          </Link>
        </div>

        <form className="space-y-4" onSubmit={handleLogin}>
          <Input
            type="text"
            placeholder="아이디"
            className="w-full h-12 bg-[#f4f4f4] border-0 rounded-md"
            value={user_id}
            onChange={(e) => setUser_id(e.target.value)}
          />
          <Input
            type="password"
            placeholder="비밀번호"
            className="w-full h-12 bg-[#f4f4f4] border-0 rounded-md"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          <button 
            type="submit" 
            style={{
              ...buttonStyle,
              ...(loginHovered ? buttonHoverStyle : {})
            }}
            onMouseEnter={() => setLoginHovered(true)}
            onMouseLeave={() => setLoginHovered(false)}
          >
            로그인
          </button>
          
          <Link 
            to="/signup" 
            style={{
              ...buttonStyle,
              ...(signupHovered ? buttonHoverStyle : {}),
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            onMouseEnter={() => setSignupHovered(true)}
            onMouseLeave={() => setSignupHovered(false)}
          >
            회원가입
          </Link>
          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center space-x-2">
            </div>
            <Link to="/find-credentials" className="text-sm text-[#8a8a8a] cursor-pointer hover:text-[#4ba6f7] transition-colors duration-200">아이디/비밀번호 찾기</Link>
          </div>
        </form>

        <div className="flex justify-center space-x-8 mt-32 text-sm text-[#8a8a8a]">
          <Link to="/terms" className="cursor-pointer hover:text-[#4ba6f7] transition-colors duration-200">이용약관</Link>
          <Link to="/privacy" className="cursor-pointer hover:text-[#4ba6f7] transition-colors duration-200">개인정보처리방침</Link>
          <Link to="/inquiry" className="cursor-pointer hover:text-[#4ba6f7] transition-colors duration-200">문의하기</Link>
        </div>
      </main>
    </div>
  );
}
