// RadioGroup.js
import React from 'react';

export function RadioGroup({ children, value, onValueChange, name }) {
  const handleChange = (event) => {
    if (onValueChange) {
      onValueChange(event.target.value);
    }
  };

  return (
    <div className="radio-group">
      {React.Children.map(children, (child) =>
        React.cloneElement(child, {
          name,
          onChange: handleChange,
          checked: child.props.value === value
        })
      )}
    </div>
  );
}

export function RadioGroupItem({ value, id, className, name, checked, onChange }) {
  return (
    <input
      type="radio"
      name={name}  // ✅ name을 props로 전달받아 설정
      value={value}
      id={id}
      className={'$className  bg-[#4ba6f7] text-white px-4 py-2 rounded-full text-sm'}
      checked={checked}
      onChange={onChange}
    />
  );
}

