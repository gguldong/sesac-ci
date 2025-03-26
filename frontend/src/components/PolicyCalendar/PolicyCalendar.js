import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import './PolicyCalendar.css';
import logo from './fundint.svg';
import useAuthStore from '../context/authStore';

const PolicyCalendar = () => {
  // 현재 날짜 정보
  const currentDate = new Date();
  const [currentYear, setCurrentYear] = useState(currentDate.getFullYear());
  const [currentMonth, setCurrentMonth] = useState(currentDate.getMonth());
  const currentDay = currentDate.getDate();

  // 정책 데이터 및 즐겨찾기 상태
  const [allPolicies, setAllPolicies] = useState([]); // 모든 정책 데이터 저장
  const [policies, setPolicies] = useState([]);
  const [events, setEvents] = useState({});
  const [policyPeriods, setPolicyPeriods] = useState({});
  const [loading, setLoading] = useState(true);
  const [dataFetched, setDataFetched] = useState(false); // 데이터 로딩 여부 추적
  
  // 상세 정보 모달을 위한 상태 추가
  const [selectedPolicy, setSelectedPolicy] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  
  // 사용자 정보 - Redux 대신 Zustand 사용
  const user = useAuthStore(state => state.user);
  const userId = user ? user.user_id : 'guest';

  const monthNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"];
  const weekdayNames = ["일", "월", "화", "수", "목", "금", "토"];

  // 정책 데이터로부터 현재 월에 해당하는 정책 및 이벤트 생성 함수
  const updateCalendarData = useCallback(() => {
    if (allPolicies.length === 0) return;
    
    // 모든 정책 표시 (필터링 없이)
    setPolicies(allPolicies);
    
    // 정책 기간 정보 및 이벤트 생성
    const periodsData = {};
    const eventsData = {};
    
    allPolicies.forEach(policy => {
      // 날짜 정보가 있는 경우에만 처리
      if (policy.startDate && policy.endDate) {
        const startDateObj = new Date(policy.startDate);
        const endDateObj = new Date(policy.endDate);
        
        // 정책 기간 정보 저장
        periodsData[policy.id] = {
          start: startDateObj,
          end: endDateObj,
          type: policy.type
        };
        
        // 현재 월에 해당하는 날짜만 달력에 하이라이트
        const currentMonthYear = new Date(currentYear, currentMonth, 1);
        const nextMonthYear = new Date(currentYear, currentMonth + 1, 0);
        
        // 정책 기간이 현재 표시 중인 월에 포함되는지 확인
        if ((startDateObj <= nextMonthYear && endDateObj >= currentMonthYear)) {
          // 현재 월에 포함되는 날짜만 이벤트로 등록
          const periodsStart = startDateObj < currentMonthYear ? currentMonthYear : startDateObj;
          const periodsEnd = endDateObj > nextMonthYear ? nextMonthYear : endDateObj;
          
          // 날짜별 이벤트 등록
          for (let day = new Date(periodsStart); day <= periodsEnd; day.setDate(day.getDate() + 1)) {
            const dateKey = `${day.getFullYear()}-${String(day.getMonth() + 1).padStart(2, '0')}-${String(day.getDate()).padStart(2, '0')}`;
            eventsData[dateKey] = {
              type: policy.type,
              name: policy.id,
              applicationType: policy.application_type
            };
          }
        }
      }
    });
    
    setPolicyPeriods(periodsData);
    setEvents(eventsData);
    setLoading(false);
  }, [allPolicies, currentMonth, currentYear]);

  // 사용자의 즐겨찾기 정책 및 정책 데이터 가져오기 - 최초 한 번만 실행
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // 이미 데이터를 가져왔다면 API 호출하지 않음
        if (dataFetched) {
          updateCalendarData();
          return;
        }
        
        // 로그인 상태 확인
        if (userId === 'guest') {
          // 비로그인 상태인 경우 샘플 데이터 표시
          const policiesData = [
            {
              id: "청년내일채움공제",
              title: "청년내일채움공제",
              content: "취업 청년의 자산형성 지원",
              type: "여유있는 정책",
              startDate: "2023-03-02",
              endDate: "2023-03-11",
              application_period: "2023년 3월 2일 ~ 3월 11일",
              application_type: "정기"
            },
            {
              id: "청년희망적금",
              title: "청년희망적금",
              content: "청년 자산형성 지원",
              type: "종료된 정책",
              startDate: "2023-03-08",
              endDate: "2023-03-17",
              application_period: "2023년 3월 8일 ~ 3월 17일",
              application_type: "정기"
            },
            {
              id: "청년주택드림",
              title: "청년주택드림",
              content: "청년 주거 지원",
              type: "마감 임박 정책",
              startDate: "2023-03-15",
              endDate: "2023-03-24",
              application_period: "2023년 3월 15일 ~ 3월 24일",
              application_type: "정기"
            },
            {
              id: "청년도약계좌",
              title: "청년도약계좌",
              content: "청년 미래 자산 형성 지원",
              type: "여유있는 정책",
              startDate: "2023-03-20",
              endDate: "2023-03-27",
              application_period: "상시 신청 가능",
              application_type: "상시"
            }
          ];
          
          console.log('비로그인 상태 - 샘플 정책 표시');
          setAllPolicies(policiesData);
          setDataFetched(true);
          updateCalendarData();
          return;
        }
        
        // 새로운 API 엔드포인트를 호출하여 즐겨찾기한 정책의 상세 정보 가져오기
        const response = await fetch(`/api/favorites/${userId}/calendar`);
        if (!response.ok) throw new Error('즐겨찾기 정책 상세 정보 가져오기 실패');
        const policiesData = await response.json();
        
        console.log('API에서 가져온 정책 데이터:', policiesData);
        
        // 모든 정책 데이터 저장
        setAllPolicies(policiesData);
        setDataFetched(true);
        updateCalendarData();
      } catch (error) {
        console.error("데이터 로딩 오류:", error);
        setLoading(false);
      }
    };
    
    fetchData();
  }, [userId, updateCalendarData, dataFetched]);
  
  // 월이 변경될 때마다 정책 데이터 업데이트
  useEffect(() => {
    if (dataFetched) {
      updateCalendarData();
    }
  }, [currentMonth, currentYear, dataFetched, updateCalendarData]);

  // 월 이동 함수
  const prevMonth = () => {
    setCurrentMonth(prev => {
      if (prev === 0) {
        setCurrentYear(prev => prev - 1);
        return 11;
      }
      return prev - 1;
    });
  };

  const nextMonth = () => {
    setCurrentMonth(prev => {
      if (prev === 11) {
        setCurrentYear(prev => prev + 1);
        return 0;
      }
      return prev + 1;
    });
  };

  // 정책 항목 하이라이트 처리
  const highlightPolicyPeriod = (policyId, highlight) => {
    const period = policyPeriods[policyId];
    if (!period) return;
    
    // 날짜 정보가 없는 경우 하이라이트하지 않음
    if (!period.start || !period.end) {
      return;
    }
    
    const startDate = new Date(period.start);
    const endDate = new Date(period.end);
    
    // 현재 표시 중인 월/년에 해당하는 날짜 하이라이트
    const currentMonthYear = new Date(currentYear, currentMonth, 1);
    const lastDayOfCurrentMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
    
    // 정책 기간이 현재 월에 포함되는지 확인
    if (!(startDate.getMonth() > currentMonth && startDate.getFullYear() >= currentYear) &&
        !(endDate.getMonth() < currentMonth && endDate.getFullYear() <= currentYear)) {
      
      const calendarDays = document.querySelectorAll('.calendar-day:not(.inactive)');
      
      calendarDays.forEach(day => {
        const dayNumber = parseInt(day.textContent);
        
        // 현재 월/년의 해당 날짜
        const currentDayDate = new Date(currentYear, currentMonth, dayNumber);
        
        // 시작일과 종료일 사이에 있는지 확인
        if (currentDayDate >= startDate && currentDayDate <= endDate) {
          if (highlight) {
            day.classList.add(`policy-${period.type.replace(/\s+/g, '-').toLowerCase()}`);
          } else {
            day.classList.remove(`policy-${period.type.replace(/\s+/g, '-').toLowerCase()}`);
          }
        }
      });
    }
  };

  // 캘린더 생성 함수
  const generateCalendar = () => {
    const firstDay = new Date(currentYear, currentMonth, 1).getDay();
    const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
    const prevMonthDays = new Date(currentYear, currentMonth, 0).getDate();
    
    const calendarDays = [];
    
    // 요일 헤더 추가
    weekdayNames.forEach(weekday => {
      calendarDays.push(
        <div key={`weekday-${weekday}`} className="calendar-weekday">
          {weekday}
        </div>
      );
    });
    
    // 날짜 생성
    let dayCounter = 1 - firstDay;
    const rowsNeeded = Math.ceil((firstDay + daysInMonth) / 7);
    const totalDays = rowsNeeded * 7;
    
    for (let i = 0; i < totalDays; i++) {
      const isCurrentDay = dayCounter === currentDay && 
                          currentYear === new Date().getFullYear() && 
                          currentMonth === new Date().getMonth();
      
      const isInactive = dayCounter <= 0 || dayCounter > daysInMonth;
      
      let dayText;
      if (dayCounter <= 0) {
        dayText = prevMonthDays + dayCounter;
      } else if (dayCounter > daysInMonth) {
        dayText = dayCounter - daysInMonth;
      } else {
        dayText = dayCounter;
      }
      
      // 날짜 클래스 설정
      let dayClass = "calendar-day";
      if (isInactive) dayClass += " inactive";
      if (isCurrentDay) dayClass += " today";
      
      // 현재 날짜 객체 생성 (현재 월/년/일)
      let currentDateObj;
      if (dayCounter <= 0) {
        // 이전 달의 날짜
        const prevMonth = currentMonth === 0 ? 11 : currentMonth - 1;
        const prevYear = currentMonth === 0 ? currentYear - 1 : currentYear;
        currentDateObj = new Date(prevYear, prevMonth, prevMonthDays + dayCounter);
      } else if (dayCounter > daysInMonth) {
        // 다음 달의 날짜
        const nextMonth = currentMonth === 11 ? 0 : currentMonth + 1;
        const nextYear = currentMonth === 11 ? currentYear + 1 : currentYear;
        currentDateObj = new Date(nextYear, nextMonth, dayCounter - daysInMonth);
      } else {
        // 현재 달의 날짜
        currentDateObj = new Date(currentYear, currentMonth, dayCounter);
      }
      
      // 날짜를 YYYY-MM-DD 형식의 문자열로 변환
      const dateKey = `${currentDateObj.getFullYear()}-${String(currentDateObj.getMonth() + 1).padStart(2, '0')}-${String(currentDateObj.getDate()).padStart(2, '0')}`;
      
      // 해당 날짜에 이벤트가 있는지 확인
      if (events[dateKey] && !isInactive) {
        const event = events[dateKey];
        const eventType = event.type;
        dayClass += ` policy-${eventType.replace(/\s+/g, '-').toLowerCase()}`;
      }
      
      calendarDays.push(
        <div 
          key={`day-${i}`} 
          className={dayClass}
        >
          {dayText}
        </div>
      );
      
      dayCounter++;
    }
    
    return calendarDays;
  };

  return (
    <div className="policy-calendar-container">
      <div className="sidebar-calendar">
        <Link to="/">
          <img src={logo} alt="펀딩인트 로고" className="fundint-logo-image" />
        </Link>
        
        <h3 className="sidebar-title">내 즐겨찾기 정책</h3>
        
        {userId === 'guest' && (
          <div className="login-notice">
            <p>로그인 후 나만의 즐겨찾기 정책을 관리할 수 있습니다.</p>
            <Link to="/login" className="login-button">로그인하기</Link>
          </div>
        )}
        
        {loading ? (
          <div className="loading">정책 정보를 불러오는 중...</div>
        ) : (
          <div className="policy-box-container">
            {policies.length === 0 ? (
              <div className="no-policies">즐겨찾기된 정책이 없습니다.</div>
            ) : (
              policies.map(policy => (
                <div 
                  key={policy.id} 
                  className={`policy-box ${policy.type.replace(/\s+/g, '-').toLowerCase()} ${policy.month === currentMonth + 1 ? 'current-month' : ''}`}
                  data-policy-id={policy.id}
                  data-type={policy.type}
                  data-application-type={policy.application_type || '일반'}
                  onMouseEnter={() => highlightPolicyPeriod(policy.id, true)}
                  onMouseLeave={() => highlightPolicyPeriod(policy.id, false)}
                  onClick={() => {
                    setSelectedPolicy(policy);
                    setShowDetailModal(true);
                  }}
                >
                  <div className="policy-title">{policy.title}</div>
                  {policy.content && <div className="policy-content">{policy.content}</div>}
                  <div className="application-period-info">
                    <span className="application-period-text">
                      {policy.application_period}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
        
        <div className="policy-info">
          <div className="policy-types-row">
            <div><span className="relaxed-icon"></span>여유있는</div>
            <div><span className="urgent-icon"></span>마감임박</div>
            <div><span className="ended-icon"></span>종료된</div>
          </div>
        </div>
      </div>
      
      <div className="calendar">
        <div className="calendar-header">
          <span id="currentMonthYear">{monthNames[currentMonth]} {currentYear}</span>
          <div className="button-container">
            <button onClick={prevMonth}>&lt;</button>
            <button onClick={nextMonth}>&gt;</button>
          </div>
        </div>
        
        <div className="calendar-days">
          {generateCalendar()}
        </div>
      </div>

      {/* 상세 정보 모달 */}
      {showDetailModal && selectedPolicy && (
        <div className="policy-calendar-modal-overlay" onClick={() => setShowDetailModal(false)}>
          <div className="policy-calendar-modal" onClick={(e) => e.stopPropagation()}>
            <div className="policy-calendar-modal-header">
              <h3>{selectedPolicy.title || '정책 상세 정보'}</h3>
              <button className="policy-calendar-modal-close-btn" onClick={() => setShowDetailModal(false)}>×</button>
            </div>
            <div className="policy-calendar-modal-content">
              <p className="policy-content policy-calendar-text-with-newlines">{selectedPolicy.content || '정책 설명'}</p>
              
              {selectedPolicy.support_type && (
                <div className="policy-support-type">
                  <span>지원 유형:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.support_type}</span>
                </div>
              )}
              {selectedPolicy.eligibility && (
                <div className="policy-eligibility">
                  <span>지원 대상:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.eligibility}</span>
                </div>
              )}
              {selectedPolicy.benefits && (
                <div className="policy-benefits">
                  <span>지원 내용:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.benefits}</span>
                </div>
              )}
              {selectedPolicy.selection_criteria && (
                <div className="policy-selection-criteria">
                  <span>선정 기준:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.selection_criteria}</span>
                </div>
              )}
              {selectedPolicy.application_period && (
                <div className="policy-application-period">
                  <span>신청 기한:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.application_period}</span>
                </div>
              )}
              {selectedPolicy.application_method && (
                <div className="policy-application-method">
                  <span>신청 방법:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.application_method}</span>
                </div>
              )}
              {selectedPolicy.required_documents && (
                <div className="policy-required-documents">
                  <span>구비 서류:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.required_documents}</span>
                </div>
              )}
              {selectedPolicy.application_office && (
                <div className="policy-application-office">
                  <span>접수 기관:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.application_office}</span>
                </div>
              )}
              {selectedPolicy.contact_info && (
                <div className="policy-contact-info">
                  <span>문의처:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.contact_info}</span>
                </div>
              )}
              {selectedPolicy.legal_basis && (
                <div className="policy-legal-basis">
                  <span>법령:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.legal_basis}</span>
                </div>
              )}
              {selectedPolicy.administrative_rule && (
                <div className="policy-administrative-rule">
                  <span>행정규칙:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.administrative_rule}</span>
                </div>
              )}
              {selectedPolicy.local_law && (
                <div className="policy-local-law">
                  <span>자치법규:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.local_law}</span>
                </div>
              )}
              {selectedPolicy.responsible_agency && (
                <div className="policy-responsible-agency">
                  <span>소관기관:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.responsible_agency}</span>
                </div>
              )}
              {selectedPolicy.last_updated && (
                <div className="policy-last-updated">
                  <span>수정일시:</span> <span className="policy-calendar-text-with-newlines">{selectedPolicy.last_updated}</span>
                </div>
              )}
              
              <div className="policy-calendar-modal-footer">
                {selectedPolicy.online_application_url ? (
                  <button 
                    className="policy-calendar-apply-button"
                    onClick={() => window.open(selectedPolicy.online_application_url, '_blank')}
                  >
                    온라인 신청하기
                  </button>
                ) : (
                  <p className="policy-calendar-no-url-hint modal-hint">(온라인 신청 불가)</p>
                )}
                <button 
                  className="policy-calendar-close-button"
                  onClick={() => setShowDetailModal(false)}
                >
                  닫기
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PolicyCalendar; 