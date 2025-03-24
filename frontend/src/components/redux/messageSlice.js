// messageSlice.js
import { createSlice } from '@reduxjs/toolkit';

const messageSlice = createSlice({
  name: 'message',
  initialState: '',
  reducers: {
    setMessage: (state, action) => action.payload,
  },
});

export const { setMessage } = messageSlice.actions;
export default messageSlice.reducer;