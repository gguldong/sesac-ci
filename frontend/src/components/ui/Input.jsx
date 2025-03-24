import React from "react";

export function Input({type, value, onChange, placeholder }) {
  return (
    <input
      type={type}    
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      className="px-3 py-2 border rounded w-full"
    />
  );
}
