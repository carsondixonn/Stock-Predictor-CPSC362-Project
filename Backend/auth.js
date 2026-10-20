// Shared auth helpers, loaded on every page.
// The backend's /login returns the user's info (no token), so we remember
// that user in localStorage to know who is "signed in" in the browser.

function getUser() {
    try {
        const raw = localStorage.getItem("user");
        return raw ? JSON.parse(raw) : null;
    } catch {
        return null;
    }
}

function setUser(user) {
    try { localStorage.setItem("user", JSON.stringify(user)); } catch {}
}

function clearUser() {
    try { localStorage.removeItem("user"); } catch {}
}

// Turns the top-right "Login" link into "Logout (username)" when signed in.
document.addEventListener("DOMContentLoaded", () => {
    const link = document.getElementById("authLink");
    const user = getUser();
    if (!link || !user) return;

    link.textContent = `Logout (${user.username})`;
    link.href = "#";
    link.classList.remove("active");
    link.addEventListener("click", (e) => {
        e.preventDefault();
        clearUser();
        window.location.href = "index.html";
    });
});