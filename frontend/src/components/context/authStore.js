import { create } from "zustand";
import axios from "axios";

const useAuthStore = create((set) => ({
  user: null,

  // 로그인 상태 확인
  fetchUser: async () => {
    try {
      const res = await axios.get("/api/me", {
        withCredentials: true,
      });
      set({ user: res.data.user });
    } catch {
      set({ user: null });
    }
  },

  // 로그인 요청
  login: async (user_id, password) => {
    try {
      const res = await axios.post(
        "/api/login",
        { user_id, password },
        { withCredentials: true }
      );
      set({ user: res.data.user });
      return true;
    } catch {
      return false;
    }
  },

  // 로그아웃 요청
  logout: async () => {
    try {
      await axios.get("/api/logout", {
        withCredentials: true,
      });
      set({ user: null });
    } catch {
      console.error("로그아웃 실패");
    }
  },
}));

export default useAuthStore;
