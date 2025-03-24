import { createSlice } from '@reduxjs/toolkit';

const sessionSlice = createSlice({
  name: 'sessions',
  initialState: [],
  reducers: {
    setSessions: (state, action) => action.payload,
  },
});

export const { setSessions } = sessionSlice.actions;
export default sessionSlice.reducer;