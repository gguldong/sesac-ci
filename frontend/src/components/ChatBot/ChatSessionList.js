/* ChatSessionList.js */
import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Link } from 'react-router-dom';
import useAuthStore from '../context/authStore';
import { setSessions } from '../redux/sessionSlice';

const ChatSessionList = () => {
  const sessions = useSelector(state => state.sessions);
  const dispatch = useDispatch();
  const user = useAuthStore.getState().user;

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const res = await fetch(`/api/sessions?user_id=${user.user_id}`);
        const data = await res.json();
        dispatch(setSessions(data)); // Redux 상태 업데이트
      } catch (error) {
        console.error("세션 목록 로딩 에러:", error);
      }
    };

    fetchSessions();
  }, [dispatch, user.user_id]);

  return (
    <div className="session-list-container">
      {/* <h2>Chat Sessions</h2> */}
      <ul className="session-list">
        {sessions.map(session => (
          <Link 
            to={`/chat/${session.sessionId}`} 
            key={session.sessionId} 
            className="session-item"
          >
            {session.header_message.substring(0, 16)}
          </Link>
        ))}
      </ul>
    </div>
  );
};

export default ChatSessionList;
