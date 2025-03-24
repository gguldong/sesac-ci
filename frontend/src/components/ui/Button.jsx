import React from "react";

export function Button({ children, onClick }) {
  return (
    <button
      onClick={onClick}
      className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
    >
      {children}
    </button>
  );
}

//회원가입 버튼
export function SignUpButton ({ values }) {
  return (
    <div className="flex justify-center gap-4 pt-4">
      {values.map((value, index) => (
        <Button
          key={index}
          variant={value === "이전" ? "outline" : "default"}
          className={`px-8 py-2 ${
            value === "이전"
              ? "border-[#bbbbbb] text-[#8a8a8a] hover:bg-gray-50"
              : "bg-[#4ba6f7] hover:bg-[#4ba6f7]/90 text-white"
          }`}
          value={value}
        >
          {value}
        </Button>
      ))}
    </div>
  );
};
