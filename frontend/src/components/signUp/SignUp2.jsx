import { useState } from "react";
import { Check } from "lucide-react";
// import { SignUpButton } from "../ui/Button"; // 사용 시 import
import { RadioGroup, RadioGroupItem } from "../ui/RadioGroup";

export default function SignUp2() {
  const [incomeRange, setIncomeRange] = useState("");
  const [personalTraits, setPersonalTraits] = useState([]);
  const [householdTraits, setHouseholdTraits] = useState([]);

  const handlePersonalTraitToggle = (trait) => {
    if (personalTraits.includes(trait)) {
      setPersonalTraits(personalTraits.filter((t) => t !== trait));
    } else {
      setPersonalTraits([...personalTraits, trait]);
    }
  };

  const handleHouseholdTraitToggle = (trait) => {
    if (householdTraits.includes(trait)) {
      setHouseholdTraits(householdTraits.filter((t) => t !== trait));
    } else {
      setHouseholdTraits([...householdTraits, trait]);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-8 py-8 font-sans">
      {/* Income Range Section */}
      <div className="mb-10">
        <div className="flex items-center gap-4 mb-6">
          <h2 className="text-lg font-medium">소득금액 구간</h2>
          <div className="bg-[#4ba6f7] text-white px-4 py-2 rounded-full text-sm">
            2025년 가구 규모별 기준중위 소득표 보기
          </div>
        </div>

        {/* 
          RadioGroup로 감싸서 
          value={incomeRange}, onValueChange={setIncomeRange}, name="incomeRange" 지정 
        */}
        <RadioGroup
          value={incomeRange}
          onValueChange={setIncomeRange}
          name="incomeRange"
        >
          <div className="flex gap-6 flex-wrap">
            {/* 0 ~ 50% */}
            <label className="flex items-center gap-2 cursor-pointer">
              <div
                className={`w-5 h-5 rounded-full flex items-center justify-center border ${
                  incomeRange === "0-50" ? "border-[#4ba6f7] bg-[#4ba6f7]" : "border-[#bbbbbb] bg-[#f4f4f4]"
                }`}
              >
                {incomeRange === "0-50" && (
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                )}
              </div>
              {/* RadioGroupItem으로 라디오 버튼 구현 */}
              <RadioGroupItem
                value="0-50"
                id="income-0-50"
                className="sr-only"
              />
              <span>0 ~ 50%</span>
            </label>

            {/* 51 ~ 75% */}
            <label className="flex items-center gap-2 cursor-pointer">
              <div
                className={`w-5 h-5 rounded-full flex items-center justify-center border ${
                  incomeRange === "51-75" ? "border-[#4ba6f7] bg-[#4ba6f7]" : "border-[#bbbbbb] bg-[#f4f4f4]"
                }`}
              >
                {incomeRange === "51-75" && (
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                )}
              </div>
              <RadioGroupItem
                value="51-75"
                id="income-51-75"
                className="sr-only"
              />
              <span>51 ~ 75%</span>
            </label>

            {/* 76 ~ 100% */}
            <label className="flex items-center gap-2 cursor-pointer">
              <div
                className={`w-5 h-5 rounded-full flex items-center justify-center border ${
                  incomeRange === "76-100" ? "border-[#4ba6f7] bg-[#4ba6f7]" : "border-[#bbbbbb] bg-[#f4f4f4]"
                }`}
              >
                {incomeRange === "76-100" && (
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                )}
              </div>
              <RadioGroupItem
                value="76-100"
                id="income-76-100"
                className="sr-only"
              />
              <span>76 ~ 100%</span>
            </label>

            {/* 101 ~ 200% */}
            <label className="flex items-center gap-2 cursor-pointer">
              <div
                className={`w-5 h-5 rounded-full flex items-center justify-center border ${
                  incomeRange === "101-200" ? "border-[#4ba6f7] bg-[#4ba6f7]" : "border-[#bbbbbb] bg-[#f4f4f4]"
                }`}
              >
                {incomeRange === "101-200" && (
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                )}
              </div>
              <RadioGroupItem
                value="101-200"
                id="income-101-200"
                className="sr-only"
              />
              <span>101 ~ 200%</span>
            </label>

            {/* 200% 이상 */}
            <label className="flex items-center gap-2 cursor-pointer">
              <div
                className={`w-5 h-5 rounded-full flex items-center justify-center border ${
                  incomeRange === "200+" ? "border-[#4ba6f7] bg-[#4ba6f7]" : "border-[#bbbbbb] bg-[#f4f4f4]"
                }`}
              >
                {incomeRange === "200+" && (
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                )}
              </div>
              <RadioGroupItem
                value="200+"
                id="income-200+"
                className="sr-only"
              />
              <span>200% 이상</span>
            </label>
          </div>
        </RadioGroup>
      </div>

      {/* Personal Characteristics Section */}
      <div className="mb-10">
        <h2 className="text-lg font-medium mb-4">개인 특성 정보 (중복 선택)</h2>
        <div className="bg-[#f4f4f4] p-6 rounded-md">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[
              "예비부부/난임",
              "임신부",
              "출산/입양",
              "장애인",
              "국가보훈대상자",
              "농업인",
              "어업인",
              "축산인",
              "임업인",
              "초등학생",
              "중학생",
              "고등학생",
              "대학생/대학원생",
              "질병/질환자",
              "구직자/실업자",
              "해당사항 없음",
            ].map((trait, index) => (
              <label key={index} className="flex items-center gap-2 cursor-pointer">
                <div
                  className={`w-5 h-5 border rounded flex items-center justify-center ${
                    personalTraits.includes(trait)
                      ? "bg-[#4ba6f7] border-[#4ba6f7]"
                      : "bg-white border-[#bbbbbb]"
                  }`}
                >
                  {personalTraits.includes(trait) && (
                    <Check className="w-3 h-3 text-white" />
                  )}
                </div>
                <input
                  type="checkbox"
                  checked={personalTraits.includes(trait)}
                  onChange={() => handlePersonalTraitToggle(trait)}
                  className="sr-only"
                />
                <span className="text-sm">{trait}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Household Characteristics Section */}
      <div className="mb-10">
        <h2 className="text-lg font-medium mb-4">가구 특성 정보</h2>
        <div className="bg-[#f4f4f4] p-6 rounded-md">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[
              "다문화가정",
              "북한이탈주민가정",
              "한부모가정/조손가정",
              "1인 가구",
              "다자녀가구",
              "무주택세대",
              "신규전입가구",
              "확대가족",
              "해당사항 없음",
            ].map((trait, index) => (
              <label key={index} className="flex items-center gap-2 cursor-pointer">
                <div
                  className={`w-5 h-5 border rounded flex items-center justify-center ${
                    householdTraits.includes(trait)
                      ? "bg-[#4ba6f7] border-[#4ba6f7]"
                      : "bg-white border-[#bbbbbb]"
                  }`}
                >
                  {householdTraits.includes(trait) && (
                    <Check className="w-3 h-3 text-white" />
                  )}
                </div>
                <input
                  type="checkbox"
                  checked={householdTraits.includes(trait)}
                  onChange={() => handleHouseholdTraitToggle(trait)}
                  className="sr-only"
                />
                <span className="text-sm">{trait}</span>
              </label>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
