# Safesight monitoring dashboard

Audience: the person supervising a live safety demonstration. The main task is
to connect a risk state to a visible person or zone and review confirmed events.

## Visual direction

A camera-first workstation with a narrow risk panel and one event table. The
camera occupies the larger area; the risk value remains alongside it on desktop
and follows it on mobile. Everything is left aligned. The table shows only the
ten most recent confirmed events.

The UI/UX skill search returned a real-time operations pattern, which fits this
task. Its glassmorphism suggestion was rejected in favor of opaque surfaces:
status and text need predictable contrast over changing camera images. There
are no decorative gradients, remote fonts, animated counters, or external assets.

Palette: paper #EDF1F4, white #FFFFFF, ink #142632, information blue #176B90;
semantic green #237454, amber #926B08, orange #B85216, red #A93036.
Bahnschrift gives the headings and numbers an industrial instrument character;
Segoe UI provides readable labels, with system sans-serif fallbacks.

## Behavior and accessibility

- Risk is expressed with a number, word, and color, never color alone.
- Simulation is identified in the header, banner, feed, events, and footer.
- Pausing freezes the display only. Monitoring and event recording continue.
- Hidden pages stop video and state polling while the monitoring worker continues.
- Missing, stale, and failed data replace the score with an unavailable state.
- State requests time out after 2.5 seconds; an old green score cannot remain
  indefinitely while a request hangs.
- The camera aspect ratio is reserved to prevent loading shifts.
- Controls have visible focus and at least 44px height; event links have explicit
  snapshot descriptions. The horizontal table region is keyboard-focusable.
- No motion effects are used; reduced-motion preferences are respected.

## Verification

Browser review at 1280px and 375px verified camera/risk layout, simulation labels,
ten event rows, and horizontal table scrolling without page overflow. Pausing
showed an explicit message that monitoring continues. No console errors were
observed. Review found and fixed a wrapped /100 label and cramped mobile table.

The local preview uses synthetic people and scripted risk conditions. Its FPS,
latency, events, and snapshots are not measurements of either detector.
