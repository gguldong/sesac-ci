import React, { useEffect, useState, useCallback } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import ChatBot from "./components/ChatBot/ChatBot.js";
import ChatSessionDetail from "./components/ChatBot/ChatSessionDetail";
import Layout from "./components/ChatBot/Layout.js";
import Header from "./components/ui/Header";
import LoginForm from "./components/signUp/LoginForm";
import SignUp0 from "./components/signUp/SignUp0";
import SignUp1 from "./components/signUp/SignUp1";
import SignUp3 from "./components/signUp/SignUp3";
import MainPage from "./components/ChatBot/MainPage";
import MyPage from "./components/mypage/MyPage";
import useAuthStore from "./components/context/authStore.js";
//redux
import { Provider } from 'react-redux';
import store from './components/redux/store'; // store.js의 경로에 맞게 수정
import "./App.css";

// 🔒 로그인 필수 페이지 보호 (PrivateRoute)
const PrivateRoute = ({ element }) => {
  const { user } = useAuthStore();
  return user ? element : <Navigate to="/login" replace />;
};

function App() {
  const { fetchUser } = useAuthStore();
  const [showHeader, setShowHeader] = useState(false);


  // 쓰로틀 함수: 지정된 시간(ms) 간격으로만 함수 실행을 허용
  const throttle = (func, delay) => {
    let lastCall = 0;
    return function(...args) {
      const now = new Date().getTime();
      if (now - lastCall < delay) {
        return;
      }
      lastCall = now;
      return func(...args);
    };
  };

  // useCallback으로 참조 안정성 보장
  const handleMouseMove = useCallback(
    throttle((e) => {
      if (e.clientY <= 50) {
        setShowHeader(true);
      } else {
        setShowHeader(false);
      }
    }, 100), // 100ms 간격으로 쓰로틀링
    []
  );

  useEffect(() => {
    fetchUser();

    // 쓰로틀링된 이벤트 핸들러 등록
    document.addEventListener('mousemove', handleMouseMove);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, [fetchUser, handleMouseMove]);

  return (
    <Provider store={store}> {/* Redux Provider로 감싸기 */}
      <Router>
        <div className={`header-container ${showHeader ? 'header-visible' : 'header-hidden'}`}>
          <Header />
        </div>
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="chat" element={<PrivateRoute element={<Layout />} />}>
            <Route index element={<ChatBot />} />
            <Route path=":sessionId" element={<ChatSessionDetail />} />
          </Route>
          <Route path="login" element={<LoginForm />} />
          <Route path="signup" element={<SignUp0 />} />
          <Route path="signup1" element={<SignUp1 />} />
          <Route path="signup3" element={<SignUp3 />} />
          <Route path="mypage" element={<PrivateRoute element={<MyPage />} />} />
        </Routes>
      </Router>
    </Provider>
  );
}

export default App;
