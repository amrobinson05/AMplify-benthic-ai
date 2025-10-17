import streamlit as st
import streamlit.components.v1 as components
import random

def render_coral_scene():
    brain_positions = [random.randint(80, 1100) for _ in range(6)]
    fan_positions = [random.randint(100, 1100) for _ in range(5)]
    anemone_positions = [random.randint(120, 1100) for _ in range(8)]
    seagrass_positions = [random.randint(40, 1150) for _ in range(35)]
    starfish_positions = [random.randint(150, 1100) for _ in range(3)]

    brain_svg = "".join(
        f"""
        <g>
          <ellipse cx="{x}" cy="280" rx="35" ry="22" fill="url(#brainCoral)" opacity="0.8"/>
          <ellipse cx="{x}" cy="280" rx="30" ry="18" fill="none" stroke="#CD5C5C" stroke-width="2" opacity="0.6"/>
          <path d="M{x-25} 280 Q{x-15} 270 {x-5} 280 Q{x+5} 270 {x+15} 280 Q{x+25} 270 {x+35} 280"
                stroke="#8B4513" stroke-width="1.2" fill="none" opacity="0.7"/>
        </g>""" for x in brain_positions
    )

    # (fan_svg, anemone_svg, seagrass_svg, starfish_svg same as before — truncated here for brevity)
    fan_svg = "".join(
        f"""<g class="seagrass" style="animation:sway-medium {4+random.random()*2:.1f}s ease-in-out infinite;">
              <path d="M{x},300 Q{x-20},230 {x-40},180 M{x},300 Q{x},240 {x},170 M{x},300 Q{x+20},230 {x+40},180"
                    stroke="url(#fanCoral)" stroke-width="3" fill="none" opacity="0.8"/>
            </g>""" for x in fan_positions
    )
    anemone_svg = "".join(
        f"""<ellipse cx="{x}" cy="295" rx="15" ry="10" fill="url(#anemone1)" opacity="0.75"
                     style="animation:anemone-pulse {3+random.random()}s ease-in-out infinite;"/>"""
        for x in anemone_positions
    )
    seagrass_svg = "".join(
        f"""<path class="seagrass" d="M{x},310 Q{x-5},250 {x},200"
              stroke="url(#{'seagrass1' if i%2==0 else 'seagrass2'})" stroke-width="3" fill="none" opacity="0.7"
              style="animation:{random.choice(['sway-gentle','sway-medium','sway-strong'])} {3+random.random()*2:.1f}s ease-in-out infinite;"/>"""
        for i, x in enumerate(seagrass_positions)
    )
    starfish_svg = "".join(
        f"""<g transform="translate({x},295) rotate({random.randint(0,45)})">
               <path d="M 0,-8 L 2,-2 L 8,-2 L 3,2 L 5,8 L 0,4 L -5,8 L -3,2 L -8,-2 L -2,-2 Z"
                     fill="#FF6347" opacity="0.85"/>
               <circle cx="0" cy="0" r="2" fill="#FF4500" opacity="0.9"/>
            </g>""" for x in starfish_positions
    )

    html = f"""
    <div style="position:fixed; bottom:0; left:0; width:100vw; height:420px; z-index:1; pointer-events:none;">
    <style>
      @keyframes sway-gentle {{0%,100%{{transform:rotate(-2deg)}}50%{{transform:rotate(2deg)}}}}
      @keyframes sway-medium {{0%,100%{{transform:rotate(-3deg)}}50%{{transform:rotate(3deg)}}}}
      @keyframes sway-strong {{0%,100%{{transform:rotate(-4deg)}}50%{{transform:rotate(4deg)}}}}
      @keyframes anemone-pulse {{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.08)}}}}
      .seagrass {{ transform-origin: bottom center; animation-iteration-count: infinite; }}
    </style>

    <svg viewBox="0 160 1200 180" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
      <defs>
        <linearGradient id="seagrass1" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#2d5016"/>
          <stop offset="50%" stop-color="#3d6e1f"/>
          <stop offset="100%" stop-color="#4a7c23"/>
        </linearGradient>
        <linearGradient id="seagrass2" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#1a4d2e"/>
          <stop offset="50%" stop-color="#2d6a3f"/>
          <stop offset="100%" stop-color="#3e8e52"/>
        </linearGradient>
        <radialGradient id="brainCoral">
          <stop offset="0%" stop-color="#FFA07A"/>
          <stop offset="50%" stop-color="#FF7F50"/>
          <stop offset="100%" stop-color="#FF6347"/>
        </radialGradient>
        <radialGradient id="fanCoral">
          <stop offset="0%" stop-color="#DA70D6"/>
          <stop offset="50%" stop-color="#BA55D3"/>
          <stop offset="100%" stop-color="#9370DB"/>
        </radialGradient>
        <radialGradient id="anemone1">
          <stop offset="0%" stop-color="#FF69B4"/>
          <stop offset="50%" stop-color="#FF1493"/>
          <stop offset="100%" stop-color="#C71585"/>
        </radialGradient>
      </defs>

      <rect x="0" y="300" width="1200" height="80" fill="#F4E4A2" opacity="0.4"/>
      <ellipse cx="300" cy="295" rx="60" ry="20" fill="#C4A77D" opacity="0.8"/>
      <ellipse cx="800" cy="300" rx="55" ry="18" fill="#BFA77B" opacity="0.8"/>

      {brain_svg}
      {fan_svg}
      {seagrass_svg}
      {anemone_svg}
      {starfish_svg}
    </svg>
    </div>
    """
    components.html(html, height=420)
