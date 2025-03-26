import React, { useState } from 'react';
import ChatSessionList from './ChatSessionList';
import { Outlet, useNavigate } from 'react-router-dom';

export default function Layout () {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true); // 사이드바 상태 관리
  const navigate = useNavigate(); // useNavigate 훅 추가

  // 사이드바 토글 함수
  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };
  
  // 새로운 채팅 버튼 클릭 시 /chat 경로로 이동
  const startNewChat = () => {
    navigate('/chat', { replace: true });
    // window.location.reload(); // 페이지 강제 새로고침
  };

  // 로고 클릭 시 루트(/) 경로로 이동하는 함수
  const navigateToHome = () => {
    navigate('/', { replace: true });
  };

  return (
    <div className="layout-container">
      {/* 사이드바에 collapsed 클래스를 추가/제거하기 */}
      <div className={`sidebar ${isSidebarOpen ? '' : 'collapsed'}`}>
      <button
        className={`sidebar-toggle ${isSidebarOpen ? 'open' : 'closed'}`}
        onClick={toggleSidebar}
      ></button>
        {isSidebarOpen && (
          <>
            <div className="sidebar-logo" onClick={navigateToHome}></div>
            <button className="new-chat-button" onClick={startNewChat}>
              새로운 채팅 +
            </button>
            <ChatSessionList />
          </>
        )}
        {!isSidebarOpen && (
          <div className="sidebar-collapsed-content">
            <button className="new-chat-button-mini" onClick={startNewChat}>
              +
            </button>
          </div>
        )}
      </div>
      <div className="main-content">
        
        <Outlet />
      </div>
    </div>
  );
}