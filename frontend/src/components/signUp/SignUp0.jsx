import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Check } from "lucide-react";

export default function SignUp0() {
  const navigate = useNavigate();

  // ▼ 입력값을 useState로 관리
  const [user_id, setUser_id] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");

  // ▼ "다음" 버튼 클릭 시
  const handleNext = (e) => {
    e.preventDefault();

    // (선택) 비밀번호 일치 여부 등 검증 가능
    if (password !== confirmPw) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }

    // ▼ 다음 페이지로 이동하면서, 입력값을 함께 전달
    navigate("/signup1", {
      state: {
        user_id,
        password,
        username,
        email,
        phone,
      },
    });
  };

  return (
    <div className="min-h-screen bg-white">
      <main className="container mx-auto px-4 max-w-md py-10">
        <form onSubmit={handleNext}>
          {/* ID Section */}
          <div className="mb-8">
            <label className="block text-gray-700 mb-2 text-lg">아이디</label>
            <div className="relative">
              <input
                type="text"
                placeholder="ID 입력"
                value={user_id}
                onChange={(e) => setUser_id(e.target.value)}
                className="w-full bg-[#f4f4f4] rounded-md py-3 px-4 text-gray-700 focus:outline-none"
              />
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2 bg-[#4ba6f7] rounded-full p-1">
                <Check className="h-4 w-4 text-white" />
              </div>
            </div>
          </div>

          {/* Password Section */}
          <div className="mb-8">
            <label className="block text-gray-700 mb-2 text-lg">비밀번호</label>
            <div className="relative mb-3">
              <input
                type="password"
                placeholder="비밀번호 입력"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-[#f4f4f4] rounded-md py-3 px-4 text-gray-700 focus:outline-none"
              />
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2 bg-[#4ba6f7] rounded-full p-1">
                <Check className="h-4 w-4 text-white" />
              </div>
            </div>
            <div className="relative mb-2">
              <input
                type="password"
                placeholder="비밀번호 재확인"
                value={confirmPw}
                onChange={(e) => setConfirmPw(e.target.value)}
                className="w-full bg-[#f4f4f4] rounded-md py-3 px-4 text-gray-700 focus:outline-none"
              />
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2 bg-[#4ba6f7] rounded-full p-1">
                <Check className="h-4 w-4 text-white" />
              </div>
            </div>
            <p className="text-[#8a8a8a] text-xs flex items-center">
              <span className="inline-block mr-1">ⓘ</span> 비밀번호는 영문, 숫자, 특수문자를
              포함하여 8자리 이상이어야 합니다.
            </p>
          </div>

          {/* Username Section */}
          <div className="mb-8">
            <label className="block text-gray-700 mb-2 text-lg">사용자 이름</label>
            <input
              type="text"
              placeholder="이름 입력"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-[#f4f4f4] rounded-md py-3 px-4 text-gray-700 focus:outline-none"
            />
          </div>

          {/* Email Section */}
          <div className="mb-8">
            <label className="block text-gray-700 mb-2 text-lg">이메일 주소</label>
            <input
              type="email"
              placeholder="예: example@gmail.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-[#f4f4f4] rounded-md py-3 px-4 text-gray-700 focus:outline-none"
            />
          </div>

          {/* Phone Section */}
          <div className="mb-8">
            <label className="block text-gray-700 mb-2 text-lg">휴대폰 전화번호</label>
            <input
              type="tel"
              placeholder="예:010-0000-0000"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              className="w-full bg-[#f4f4f4] rounded-md py-3 px-4 text-gray-700 focus:outline-none mb-2"
            />
            <p className="text-[#8a8a8a] text-xs">
              등록하신 인증정보를 이용하여 아이디, 비밀번호를 찾을 수 있으며, 알림을 받습니다.
            </p>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            className="w-full bg-[#4ba6f7] text-white py-4 rounded-md font-medium text-lg hover:bg-[#3a95e6] transition-colors"
          >
            다음
          </button>
        </form>
      </main>
    </div>
  );
}
