// components/ui/checkbox.js
import React from 'react';

export const Checkbox = ({ id, checked, onChange }) => {
  return (
    <input
      type="checkbox"
      id={id}
      checked={checked}
      onChange={onChange}
      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
    />
  );
};
