// ChatSessionDetail.js
import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { useSelector, useDispatch } from 'react-redux';
import './ChatBot.css';
import { handleMicrophoneClick } from '../utils/microphoneHandler';
import { setMessage } from '../redux/messageSlice';
import useAuthStore from '../context/authStore';

const ChatSessionDetail = () => {
  const { sessionId } = useParams(); // URL에서 sessionId를 추출합니다.
  const [chatHistory, setChatHistory] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const messagesEndRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false); // 로딩 상태 추가
  const dispatch = useDispatch();
  const message = useSelector(state => state.message);
  
  // 정책 데이터를 저장할 상태 추가
  const [policies, setPolicies] = useState({});
  
  // 상세 정보 모달을 위한 상태 추가
  const [selectedPolicy, setSelectedPolicy] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  // 즐겨찾기 상태 관리 
  const [favorites, setFavorites] = useState([]);
  // useAuthStore에서 사용자 정보 가져오기
  const user = useAuthStore.getState().user;
  const userId = user ? user.user_id : 'guest'; // 로그인 여부에 따라 실제 user_id 또는 'guest' 사용

  // 백엔드에서 해당 세션의 채팅 기록을 불러옵니다.
  useEffect(() => {
    console.log("useEffect");
    fetch(`/api/sessions/${sessionId}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.messages && data.messages.length > 0) {
          // 타임스탬프와 sender에 따라 메시지를 정렬합니다.
          console.log("메시지 기록 있음");
          const sortedMessages = data.messages.sort((a, b) => {
            return a.timestamp.localeCompare(b.timestamp) || (a.sender === 'user' ? -1 : 1);
          });

          // 이미 백엔드에서 생성된 메시지 ID 사용
          setChatHistory(sortedMessages);

          // 정책 정보가 있는 메시지 찾기
          const policiesData = {};
          sortedMessages.forEach(msg => {
            // policies 필드가 있는지 확인하고 message_id 필드를 확인
            if (msg.policies && msg.message_id) {
              // 각 정책 항목의 format 일관성 확보 및 모든 상세 정보 포함
              const formattedPolicies = msg.policies.map(policy => ({
                // 필수 식별 필드
                id: policy.id || policy.service_id || "",  // 원본 서비스ID 우선 사용
                service_id: policy.service_id || policy.id || "",  // 서비스ID 필드 사용
                
                // 기본 정보
                title: policy.title || "",
                content: policy.content || policy.description || "",
                support_type: policy.support_type || "",
                
                // 지원 대상 및 내용
                eligibility: policy.eligibility || "",
                benefits: policy.benefits || "",
                selection_criteria: policy.selection_criteria || "",
                
                // 신청 관련 정보
                application_period: policy.application_period || "",
                application_method: policy.application_method || "",
                required_documents: policy.required_documents || "",
                application_office: policy.application_office || "",
                
                // 문의 및 법령 정보
                contact_info: policy.contact_info || "",
                legal_basis: policy.legal_basis || "",
                administrative_rule: policy.administrative_rule || "",  // 추가: 행정규칙
                local_law: policy.local_law || "",  // 추가: 자치법규
                
                // 기관 및 상태 정보
                responsible_agency: policy.responsible_agency || "",  // 추가: 소관기관명
                last_updated: policy.last_updated || "",  // 추가: 수정일시
                
                // 링크
                online_application_url: policy.online_application_url || "",
  
              }));
              
              policiesData[msg.message_id] = formattedPolicies;
            }
          });
          
          if (Object.keys(policiesData).length > 0) {
            setPolicies(policiesData);
          }
        } 
        else {
          if (message) {
            // 메시지를 처리하는 로직
            console.log("처리할 메시지:", message);
            sendMessage(message);
            dispatch(setMessage('')); // Redux 메시지 초기화
          }
          else {
            console.log("처리할 메시지 없음");
          }
        } 
      })
      .catch((err) => console.error(err));
  }, [sessionId, message, dispatch]);

  // 컴포넌트 마운트 시 사용자의 즐겨찾기 목록 가져오기
  useEffect(() => {
    fetchUserFavorites();
  }, [userId]); // userId가 변경될 때마다 즐겨찾기 목록 다시 가져오기

  // 사용자의 즐겨찾기 목록 가져오기
  const fetchUserFavorites = async () => {
    try {
      const response = await fetch(`/api/favorites/${userId}`);
      if (response.ok) {
        const data = await response.json();
        // 즐겨찾기된 정책 ID 목록 추출
        const favoriteIds = data.map(fav => fav.policy_id);
        setFavorites(favoriteIds);
      }
    } catch (error) {
      console.error("즐겨찾기 목록 가져오기 오류:", error);
    }
  };

  // 즐겨찾기 추가/제거 함수
  const toggleFavorite = async (policyId) => {
    try {
      if (favorites.includes(policyId)) {
        // 즐겨찾기 제거
        const response = await fetch(`/api/favorites/${userId}/${policyId}`, {
          method: 'DELETE'
        });
        
        if (response.ok) {
          setFavorites(prev => prev.filter(id => id !== policyId));
        }
      } else {
        // 즐겨찾기 추가
        const response = await fetch('/api/favorites', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            policy_id: policyId
          })
        });
        
        if (response.ok) {
          setFavorites(prev => [...prev, policyId]);
        }
      }
    } catch (error) {
      console.error("즐겨찾기 토글 오류:", error);
    }
  };

  // 메시지 전송 함수: 백엔드의 /chat 엔드포인트를 호출해 챗봇 응답을 받고,
  // 사용자와 챗봇 메시지를 백엔드에 저장합니다.
  const sendMessage = async (message) => {
    try {
      const timestamp = new Date().toISOString();
      const msg_user = { timestamp, sender: 'user', message: message };
      
      // 먼저 사용자 메시지를 화면에 추가 (임시 ID로 추가)
      const tempId = `temp_${Date.now()}`;
      const tempUserMsg = { ...msg_user, message_id: tempId };
      setChatHistory((prev) => [...prev, tempUserMsg]);
      
      // 로딩 상태 활성화
      setIsLoading(true);
      
      // 백엔드에 사용자 메시지 저장
      const userMsgResponse = await fetch(`/api/sessions/${sessionId}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(msg_user),
      });
      
      // 백엔드로부터 메시지 ID 받기
      const userMsgData = await userMsgResponse.json();
      
      // 사용자 메시지 ID 업데이트 (백엔드에서 생성된 ID로)
      if (userMsgData && userMsgData.message_id) {
        setChatHistory(prev => 
          prev.map(msg => 
            msg.message_id === tempId 
              ? { ...msg, message_id: userMsgData.message_id } 
              : msg
          )
        );
      }
      
      // 챗봇 응답 요청
      const response = await fetch('/api/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
           message,
           session_id: sessionId,
          }),
      });
      const data = await response.json();
      
      // 정책 데이터가 있는지 확인 (백엔드 응답 형식에 맞춤)
      const hasPolicies = data.policies && Array.isArray(data.policies) && data.policies.length > 0;
      
      const msg_bot = { 
        timestamp: new Date().toISOString(), 
        sender: 'bot', 
        message: data.response
      };
      
      // 백엔드에 챗봇 응답 저장 (정책 정보 포함)
      const botMsgResponse = await fetch(`/api/sessions/${sessionId}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...msg_bot,
          policies: hasPolicies ? data.policies : undefined
        }),
      });
      
      // 백엔드로부터 메시지 ID 받기
      const botMsgData = await botMsgResponse.json();
      
      // 백엔드에서 반환한 message_id 추가
      if (botMsgData && botMsgData.message_id) {
        msg_bot.message_id = botMsgData.message_id;
        
        // 정책 데이터가 있으면 상태에 저장
        if (hasPolicies) {
          // 정책 정보의 구조가 backend에서 제공하는 형식과 일치하는지 확인
          const formattedPolicies = data.policies.map(policy => ({
            // 필수 식별 필드
            id: policy.id || policy.service_id || "",  // 원본 서비스ID 우선 사용
            service_id: policy.service_id || policy.id || "",  // 서비스ID 필드 사용
            
            // 기본 정보
            title: policy.title || "",
            content: policy.content || policy.description || "",
            support_type: policy.support_type || "",
            
            // 지원 대상 및 내용
            eligibility: policy.eligibility || "",
            benefits: policy.benefits || "",
            selection_criteria: policy.selection_criteria || "",
            
            // 신청 관련 정보
            application_period: policy.application_period || "",
            application_method: policy.application_method || "",
            required_documents: policy.required_documents || "",
            application_office: policy.application_office || "",
            contact_info: policy.contact_info || "",
            legal_basis: policy.legal_basis || "",
            online_application_url: policy.online_application_url || "",
            administrative_rule: policy.administrative_rule || "",  // 추가: 행정규칙
            local_law: policy.local_law || "",  // 추가: 자치법규
            
            // 기관 및 상태 정보
            responsible_agency: policy.responsible_agency || "",  // 추가: 소관기관명
            last_updated: policy.last_updated || "",  // 추가: 수정일시
          }));
          
          setPolicies(prev => ({
            ...prev,
            [botMsgData.message_id]: formattedPolicies
          }));
        }
      }
  
      // 챗봇 응답을 화면에 추가
      setChatHistory((prev) => [...prev.filter(msg => msg.sender !== 'loading'), msg_bot]);
  
    } catch (error) {
      console.error("메시지 전송 에러:", error);
      // 에러 발생 시 로딩 표시 제거
      setChatHistory((prev) => prev.filter(msg => msg.sender !== 'loading'));
    } finally {
      // 로딩 상태 비활성화
      setIsLoading(false);
    }
  };
  
  // 전송 버튼 클릭 시 처리
  const handleSend = () => {
    if (!currentMessage.trim()) return;
    sendMessage(currentMessage);
    setCurrentMessage('');
  };

  const localHandleMicrophoneClick = () => {
    setIsRecording(prev => !prev);
    handleMicrophoneClick(setCurrentMessage, setIsRecording);
  };

  // 새로운 메시지가 추가되면 스크롤을 자동으로 맨 아래로 이동
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  return (
    <div className="chatbot-container">
      <div className="messages">
        {chatHistory.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
                <div className="message-container">
                    <ReactMarkdown>{msg.message}</ReactMarkdown>
                    
                    {/* 봇 메시지이고 정책 목록이 있을 경우 정책 목록 UI 표시 */}
                    {msg.sender === 'bot' && msg.message_id && policies[msg.message_id] && (
                      <div className="policy-list-container">
                        <h4>추천 정책 목록</h4>
                        <div className="policy-list">
                          {policies[msg.message_id].map((policy, pIndex) => (
                            <div 
                              key={pIndex} 
                              className={`policy-item ${policy.online_application_url ? 'clickable' : 'not-clickable'}`}
                              onClick={(e) => {
                                if (policy.online_application_url) {
                                  window.open(policy.online_application_url, '_blank');
                                }
                              }}
                              data-index={pIndex}
                            >
                              <div className="policy-header">
                                <h5>{policy.title || '정책명'}</h5>
                                <button 
                                  className="favorite-button" 
                                  onClick={(e) => {
                                    e.stopPropagation(); // 이벤트 버블링 방지
                                    // service_id 또는 id를 우선 사용, title은 사용하지 않음
                                    const policyId = policy.service_id || policy.id;
                                    // ID가 존재하는 경우에만 즐겨찾기 토글 수행
                                    if (policyId) {
                                      toggleFavorite(policyId);
                                    } else {
                                      // ID가 없는 경우 알림 (선택적)
                                      console.warn("정책 ID가 없어 즐겨찾기를 저장할 수 없습니다.");
                                    }
                                  }}
                                >
                                  {favorites.includes(policy.service_id) || favorites.includes(policy.id) ? 
                                    <span className="favorite-icon active">★</span> : 
                                    <span className="favorite-icon">☆</span>
                                  }
                                </button>
                              </div>
                              <p className="policy-id" style={{ fontSize: '12px', color: '#777', marginTop: '2px', marginBottom: '5px' }}>
                                정책 ID: {policy.service_id || policy.id || '정보 없음'}
                              </p>
                              <p className="policy-content">{policy.content || '정책 설명'}</p>
                              {policy.eligibility && (
                                <div className="policy-eligibility">
                                  <span>지원 대상:</span> <span className="text-with-newlines">{policy.eligibility}</span>
                                </div>
                              )}
                              {policy.benefits && (
                                <div className="policy-benefits">
                                  <span>지원 내용:</span> <span className="text-with-newlines">{policy.benefits}</span>
                                </div>
                              )}
                              
                              <button 
                                className="detail-button"
                                onClick={(e) => {
                                  e.stopPropagation(); // 이벤트 버블링 방지
                                  setSelectedPolicy(policy);
                                  setShowDetailModal(true);
                                }}
                              >
                                상세 정보 보기
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                </div>
            </div>
        ))}
        {isLoading && (
          <div className="message bot loading">
            <div className="message-container">
              <div className="loading-indicator">
                <div className="dot dot1"></div>
                <div className="dot dot2"></div>
                <div className="dot dot3"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 상세 정보 모달 */}
      {showDetailModal && selectedPolicy && (
        <div className="policy-detail-modal-overlay" onClick={() => setShowDetailModal(false)}>
          <div className="policy-detail-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{selectedPolicy.title || '정책 상세 정보'}</h3>
              <button className="modal-close-btn" onClick={() => setShowDetailModal(false)}>×</button>
            </div>
            <div className="modal-content">
              <p className="policy-id" style={{ fontSize: '13px', color: '#777', marginTop: '0px', marginBottom: '10px' }}>
                정책 ID: {selectedPolicy.service_id || selectedPolicy.id || '정보 없음'}
              </p>
              <p className="policy-content text-with-newlines">{selectedPolicy.content || '정책 설명'}</p>
              
              {selectedPolicy.support_type && (
                <div className="policy-support-type">
                  <span>지원 유형:</span> <span className="text-with-newlines">{selectedPolicy.support_type}</span>
                </div>
              )}
              {selectedPolicy.eligibility && (
                <div className="policy-eligibility">
                  <span>지원 대상:</span> <span className="text-with-newlines">{selectedPolicy.eligibility}</span>
                </div>
              )}
              {selectedPolicy.benefits && (
                <div className="policy-benefits">
                  <span>지원 내용:</span> <span className="text-with-newlines">{selectedPolicy.benefits}</span>
                </div>
              )}
              {selectedPolicy.selection_criteria && (
                <div className="policy-selection-criteria">
                  <span>선정 기준:</span> <span className="text-with-newlines">{selectedPolicy.selection_criteria}</span>
                </div>
              )}
              {selectedPolicy.application_period && (
                <div className="policy-application-period">
                  <span>신청 기한:</span> <span className="text-with-newlines">{selectedPolicy.application_period}</span>
                </div>
              )}
              {selectedPolicy.application_method && (
                <div className="policy-application-method">
                  <span>신청 방법:</span> <span className="text-with-newlines">{selectedPolicy.application_method}</span>
                </div>
              )}
              {selectedPolicy.required_documents && (
                <div className="policy-required-documents">
                  <span>구비 서류:</span> <span className="text-with-newlines">{selectedPolicy.required_documents}</span>
                </div>
              )}
              {selectedPolicy.application_office && (
                <div className="policy-application-office">
                  <span>접수 기관:</span> <span className="text-with-newlines">{selectedPolicy.application_office}</span>
                </div>
              )}
              {selectedPolicy.contact_info && (
                <div className="policy-contact-info">
                  <span>문의처:</span> <span className="text-with-newlines">{selectedPolicy.contact_info}</span>
                </div>
              )}
              {selectedPolicy.legal_basis && (
                <div className="policy-legal-basis">
                  <span>법령:</span> <span className="text-with-newlines">{selectedPolicy.legal_basis}</span>
                </div>
              )}
              {/* 새로 추가된 필드들 */}
              {selectedPolicy.administrative_rule && (
                <div className="policy-administrative-rule">
                  <span>행정규칙:</span> <span className="text-with-newlines">{selectedPolicy.administrative_rule}</span>
                </div>
              )}
              {selectedPolicy.local_law && (
                <div className="policy-local-law">
                  <span>자치법규:</span> <span className="text-with-newlines">{selectedPolicy.local_law}</span>
                </div>
              )}
              {selectedPolicy.responsible_agency && (
                <div className="policy-responsible-agency">
                  <span>소관기관:</span> <span className="text-with-newlines">{selectedPolicy.responsible_agency}</span>
                </div>
              )}
              {selectedPolicy.last_updated && (
                <div className="policy-last-updated">
                  <span>수정일시:</span> <span className="text-with-newlines">{selectedPolicy.last_updated}</span>
                </div>
              )}
              
              <div className="modal-footer">
                {selectedPolicy.online_application_url ? (
                  <button 
                    className="apply-button"
                    onClick={() => window.open(selectedPolicy.online_application_url, '_blank')}
                  >
                    온라인 신청하기
                  </button>
                ) : (
                  <p className="no-url-hint modal-hint">(온라인 신청 불가)</p>
                )}
                <button 
                  className="close-button"
                  onClick={() => setShowDetailModal(false)}
                >
                  닫기
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      
      <div className={`input-container ${'chat-started'} ${isRecording ? 'recording' : ''}`}>
        <input
          type="text"
          placeholder="메시지를 입력하세요..."
          value={currentMessage}
          onChange={(e) => setCurrentMessage(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
              e.preventDefault(); // 기본 동작(예: 폼 제출) 방지
              if (!e.repeat) {    // 키 이벤트가 반복되지 않은 경우에만 처리
                handleSend();
              }
            }
          }}
        />
        <button className="send-button" onClick={handleSend}>전송</button>
        <button 
          className={`microphone-button ${isRecording ? 'recording' : ''}`} 
          onClick={localHandleMicrophoneClick}
        ></button>      
      </div>
    </div>
  );
};

export default ChatSessionDetail;
