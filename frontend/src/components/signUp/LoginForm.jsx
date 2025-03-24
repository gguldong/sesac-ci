import React, { useState } from "react";
//import { Checkbox } from "../ui/Checkbox";
import { Input } from "../ui/Input";
import { Link, useNavigate } from "react-router-dom";
import '../../output.css';
//import axios from "axios";
import useAuthStore from "../context/authStore.js";
// LoginForm component
export default function LoginForm() {
  const { login } = useAuthStore();
  const [user_id, setUser_id] = useState("");
  const [password, setPassword] = useState("");
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
  
  return (
    <div className="min-h-screen bg-white">
      <main className="max-w-md mx-auto mt-20 px-4">
        <div className="text-center mb-16">
          <h1 className="text-[#4ba6f7] text-6xl font-bold">Fundit</h1>
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

          <button type="submit" className="w-full h-12 bg-[#4ba6f7] text-white rounded-md font-medium">
            로그인
          </button>
          <Link to="/signup" className="w-full h-12 bg-[#4ba6f7] text-white rounded-md font-medium flex items-center justify-center">
            회원가입
          </Link>
          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center space-x-2">
            </div>
            <Link to="/find-credentials" className="text-sm text-[#8a8a8a]">아이디/비밀번호 찾기</Link>
          </div>
        </form>

        <div className="flex justify-center space-x-8 mt-32 text-sm text-[#8a8a8a]">
          <Link to="/terms">이용약관</Link>
          <Link to="/privacy">개인정보처리방침</Link>
          <Link to="/inquiry">문의하기</Link>
        </div>
      </main>
    </div>
  );
}
