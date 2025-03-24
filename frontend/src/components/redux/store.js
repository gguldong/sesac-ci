import { configureStore } from '@reduxjs/toolkit';
import sessionReducer from './sessionSlice';
import messageReducer from './messageSlice';

const store = configureStore({
  reducer: {
    sessions: sessionReducer,
    message: messageReducer,
  },
});

export default store;