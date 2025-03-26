import React from 'react';
import './PolicyCalendar.css';

/**
 * 정책의 신청기한 유형을 시각적으로 표시하는 배지 컴포넌트
 * @param {Object} props
 * @param {string} props.type - 신청기한 유형 ("상시", "조건부", "정기", "규정참조" 등)
 * @param {string} props.period - 원본 신청기한 텍스트 (선택적)
 * @param {boolean} props.showText - 텍스트 표시 여부 (기본값: true)
 */
const ApplicationPeriodBadge = ({ type, period, showText = true }) => {
  // 유형에 따른 표시 텍스트와 스타일 결정
  let displayText = '';
  let badgeClass = '';
  
  switch (type) {
    case '상시':
      displayText = '상시 신청';
      badgeClass = 'badge-permanent';
      break;
    case '조건부':
      displayText = '조건부 신청';
      badgeClass = 'badge-conditional';
      break;
    case '정기':
      displayText = '정기 신청';
      badgeClass = 'badge-regular';
      break;
    case '규정참조':
      displayText = '규정 참조';
      badgeClass = 'badge-regulation';
      break;
    default:
      if (period && period.includes('~')) {
        displayText = '기간 신청';
        badgeClass = 'badge-period';
      } else {
        displayText = '일반 신청';
        badgeClass = 'badge-default';
      }
  }
  
  return (
    <div className={`application-period-badge ${badgeClass}`}>
      {showText && displayText}
    </div>
  );
};

export default ApplicationPeriodBadge; 