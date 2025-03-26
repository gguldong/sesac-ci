import React from 'react';
import { ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import '../../output.css';
import useAuthStore from "../context/authStore.js";


const MyPage = () => {
  const { user } = useAuthStore();
  // 생년월일(birthDate)을 기반으로 나이 계산하는 함수
  const calculateAge = (birthDateStr) => {
    const birthDate = new Date(birthDateStr);
    const today = new Date();
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    return age;
  };
  return (
    <div>
      {user ? (
        <div className="min-h-screen bg-white">
          {/* Main Content */}
          <main className="container mx-auto px-4 py-12">
            {/* User Profile Header */}
            <h1 className="text-[#00316b] text-3xl font-bold mb-8">
              {user.username}님 ({user.birthDate ? calculateAge(user.birthDate) : "생년월일 정보 없음"}세)
            </h1>
            {/* Stats Card */}
            <div className="bg-[#f4f4f4] rounded-lg p-8 mb-16">
              <h2 className="text-[#00316b] text-xl font-bold mb-8">나의 혜택 조회하기</h2>

              <div className="flex">
                {/* Stats Section */}
                <div className="flex-1 flex space-x-8 pr-8 border-r border-[#bbbbbb]">
                  {/* Stat 1 */}
                  <div className="flex flex-col items-center">
                    <div className="bg-white rounded-full w-28 h-28 flex items-center justify-center shadow-md mb-4">
                      <span className="text-primary text-5xl font-bold">0</span>
                    </div>
                    <span className="text-[#00316b] text-sm">신청 가능한 혜택</span>
                  </div>

                  {/* Stat 2 */}
                  <div className="flex flex-col items-center">
                    <div className="bg-white rounded-full w-28 h-28 flex items-center justify-center shadow-md mb-4">
                      <span className="text-primary text-5xl font-bold">32</span>
                    </div>
                    <span className="text-[#00316b] text-sm">확인 필요한 혜택</span>
                  </div>

                  {/* Stat 3 */}
                  <div className="flex flex-col items-center">
                    <div className="bg-white rounded-full w-28 h-28 flex items-center justify-center shadow-md mb-4">
                      <span className="text-primary text-5xl font-bold">0</span>
                    </div>
                    <span className="text-[#00316b] text-sm">받고 있는 혜택</span>
                  </div>
                </div>

                {/* Menu Section */}
                <div className="w-64 pl-8 flex flex-col space-y-6">
                  <Link to="#" className="flex items-center justify-between text-[#00316b] hover:text-primary">
                    <span>가족 등록/관리</span>
                    <ChevronRight className="h-5 w-5" />
                  </Link>
                  <Link to="#" className="flex items-center justify-between text-[#00316b] hover:text-primary">
                    <span>개인/가구 특성관리</span>
                    <ChevronRight className="h-5 w-5" />
                  </Link>
                </div>
              </div>
            </div>
          </main>
        </div>
      ) : (
        <p>로그인이 필요합니다.</p>
      )}
    </div>
  );
};

export default MyPage;
