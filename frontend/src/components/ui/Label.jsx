// Label.js
import React from 'react';

export function Label({ htmlFor, children, className }) {
  return (
    <label htmlFor={htmlFor} className={className}>
      {children}
    </label>
  );
}
