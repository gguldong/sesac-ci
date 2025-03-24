
/* Create the database */
CREATE DATABASE  IF NOT EXISTS final_project;

/* Switch to the classicmodels database */
USE final_project;

/* Drop existing tables 
DROP TABLE IF EXISTS result;
 */
 
CREATE TABLE user (
  user_id varchar(20) NOT NULL,
  password varchar(255) DEFAULT NULL,
  username varchar(20) DEFAULT NULL,
  email varchar(100) DEFAULT NULL,
  phone varchar(20) DEFAULT NULL,
  area varchar(255) DEFAULT NULL,
  district varchar(255) DEFAULT NULL,
  birthDate date DEFAULT NULL,
  gender varchar(10) DEFAULT NULL,
  incomeRange varchar(50) DEFAULT NULL,
  personalCharacteristics varchar(255) DEFAULT NULL,
  householdCharacteristics varchar(255) DEFAULT NULL,
  PRIMARY KEY (user_id),
  KEY ix_user_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE benefits (
    id INT AUTO_INCREMENT PRIMARY KEY,
    area VARCHAR(50) NOT NULL,  -- 지역 정보
    district VARCHAR(100),      -- 구/군 정보
    min_age INT,                -- 최소 연령
    max_age INT,                -- 최대 연령
    age_summary VARCHAR(255),   -- 연령 요약
    gender VARCHAR(10),         -- 성별
    income_category VARCHAR(50),-- 소득 카테고리
    income_summary VARCHAR(255),-- 소득 요약
    personal_category VARCHAR(255), -- 개인 카테고리
    personal_summary TEXT,      -- 개인 요약
    household_category VARCHAR(255), -- 가구 카테고리
    household_summary TEXT,     -- 가구 요약
    support_type VARCHAR(50),   -- 지원 유형
    support_summary TEXT,       -- 지원 요약
    application_method VARCHAR(100), -- 신청 방법
    application_summary TEXT,   -- 신청 요약
    benefit_category VARCHAR(100), -- 혜택 카테고리
    benefit_summary TEXT,       -- 혜택 요약
    start_date DATE,            -- 시작 날짜
    end_date DATE,              -- 종료 날짜
    date_summary VARCHAR(255),  -- 날짜 요약
    benefit_details TEXT,       -- 혜택 세부사항
    source VARCHAR(255),        -- 출처
    additional_data VARCHAR(10),-- 추가 데이터
    keywords TEXT,              -- 키워드
    service_id VARCHAR(50) UNIQUE -- 서비스 ID
);

/* Create the tables */
-- chat_session 테이블 생성
CREATE TABLE chat_session (
    session_id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    header_message VARCHAR(255) DEFAULT 'Default Header',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user(user_id)
);

-- chat_message 테이블 생성
CREATE TABLE chat_message (
    message_id VARCHAR(100) PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    sender VARCHAR(20),
    message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_session(session_id)
);
