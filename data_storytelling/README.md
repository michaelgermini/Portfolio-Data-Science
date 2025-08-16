# Data Storytelling – Notes and Patterns

Reusable ideas to enrich narrative and UX across dashboards. Use this as a checklist when adding new pages.

## Principles
- Focus on one message per view. Reduce clutter; highlight the “why”.
- Lead with a headline insight; support it with a concise caption.
- Show uncertainty and caveats (don’t overstate causality).

## Narrative building blocks
- Title (insight-oriented): “CO₂ intensity fell 35% since 1990, led by X and Y.”
- Subtitle: context, scope, and time window.
- Body: 2–4 bullet points that explain the drivers and exceptions.
- Call-to-action: what the viewer should do next (filter, compare, download).

## Annotations & callouts
- Label peaks, troughs, breaks-in-series, policy changes, or major events.
- Prefer short, action-style annotations: “Policy introduced”, “Method change”.
- Keep 3–5 annotations per chart; avoid annotation overload.

Example (Plotly):
```python
fig.add_annotation(x=policy_year, y=series_value,
                   text="Policy introduced",
                   showarrow=True, arrowhead=2, ax=20, ay=-30)
```

## Glossary (short definitions)
- GDP per capita: Output per person, inflation-adjusted unless specified.
- CO₂ emissions: Annual territorial emissions of carbon dioxide.
- AQI: Air Quality Index; higher is worse.
- R²: Share of variance explained by the model (0–1).

Place glossary near charts that use specialized terms; link to it from captions.

## Sources & caveats
- Source line under each chart: dataset + last update date if known.
- Caveats: measurement changes, missing countries/years, modeling assumptions.
- Link to raw data and code where possible.

Example caption:
> Source: Our World in Data (downloaded 2025‑08‑01). Caveat: energy_per_capita missing pre‑1970 for some regions.

## Color & accessibility
- Use color‑blind friendly palettes (Okabe‑Ito recommended):
  - #000000 (black), #E69F00 (orange), #56B4E9 (sky blue), #009E73 (bluish green),
    #F0E442 (yellow), #0072B2 (blue), #D55E00 (vermillion), #CC79A7 (reddish purple)
- Reserve one strong accent color for the focal series; desaturate others.
- Always pair color with another cue (markers, dashes, ordering) for accessibility.

## Layout & mobile
- Use 1–2 key charts per screen; keep legends compact.
- Prefer responsive components and avoid wide tables on mobile.
- Put controls (filters) in the sidebar; keep top area for headlines.

## Performance
- Cache expensive data loads and computations.
- Downsample large time series for overview; provide detail on demand.
- Avoid rendering more than ~5K SVG points in a single chart if possible.

## QA checklist before publishing
- Axes have units; titles state the timeframe and scope.
- Colors are consistent across related charts.
- Annotations don’t overlap points or axes on common screen sizes.
- Source and caveats are present; links work.
- Numbers are formatted with thousands separators and appropriate precision.

## Reusable snippets
- Plotly reference line:
```python
fig.add_shape(type="line", x0=xmin, y0=ymin, x1=xmax, y1=ymax,
              line=dict(color="#999", dash="dash"))
```
- Folium circle markers sized by magnitude:
```python
folium.CircleMarker(location=(lat, lon), radius=max(5, value**0.5),
                    color=color, fill=True, fill_opacity=0.6).add_to(m)
```

## Where to place notes
- App page: “Storytelling notes” (overview bullets).
- Repo file: `data_storytelling/README.md` (this document) – deeper guidance and examples.

Components and ideas to enhance dashboard storytelling (annotations, callouts, glossary, source links).

Reusable in `exploration_inegalites` and `climat_energie`.



