import React from "react";
import "./Landing.css";

export default function Landing({ onStart }) {
  return (
    <div className="landing-root">
      <div className="landing-card">
        <div className="landing-hero">
          {/* simple illustrative SVG */}
          <svg viewBox="0 0 800 400" className="landing-svg" xmlns="http://www.w3.org/2000/svg" aria-hidden>
            <defs>
              <linearGradient id="g1" x1="0" x2="1">
                <stop offset="0" stopColor="#6EE7B7" />
                <stop offset="1" stopColor="#60A5FA" />
              </linearGradient>
            </defs>

            <rect x="0" y="0" width="800" height="400" fill="#071026" rx="12"/>

            <g transform="translate(40,40)">
              <rect x="0" y="20" width="420" height="220" rx="10" fill="url(#g1)" opacity="0.12"/>
              <g transform="translate(20,40)">
                {/* charts */}
                <polyline points="0,120 40,80 80,100 120,60 160,70 200,30 240,50" fill="none" stroke="#60A5FA" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="40" cy="80" r="6" fill="#60A5FA" />
                <circle cx="120" cy="60" r="6" fill="#34D399" />
                <rect x="260" y="30" width="90" height="90" rx="8" fill="#0b1220" stroke="#2b3440" />
                <text x="275" y="80" fill="#cfe8ff" fontSize="14">DAU</text>
              </g>
              <g transform="translate(470,40)">
                <rect x="0" y="0" width="220" height="140" rx="8" fill="#071026" stroke="#173046" />
                <g transform="translate(18,18)" fill="#9ccaff">
                  <rect x="0" y="72" width="16" height="48" rx="3"/>
                  <rect x="28" y="40" width="16" height="80" rx="3"/>
                  <rect x="56" y="16" width="16" height="104" rx="3"/>
                </g>
              </g>
            </g>
          </svg>
        </div>

        <h1 className="landing-title">User Engagement Forecast Tool</h1>
        <p className="landing-sub">
          Upload your usage logs (CSV) and generate short-term forecasts of Daily Active Users using the hybrid models.
        </p>

        <div className="landing-actions">
          <button className="btn-primary" onClick={onStart}>Get started</button>
          <a className="btn-ghost" href="#" onClick={(e)=>{ e.preventDefault(); alert("Documentation coming soon"); }}>
            How it works
          </a>
        </div>
      </div>
    </div>
  );
}