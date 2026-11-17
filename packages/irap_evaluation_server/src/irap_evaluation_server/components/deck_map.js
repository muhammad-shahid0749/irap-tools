// A Vue component for NiceGUI: points on a MapLibre basemap, drawn by deck.gl (see deck_map.py).
// Python sends the outcomes of the points with `set_outcomes`, the highlighted points with
// `set_highlighted` and the selected point with `set_selected`, and receives clicks on points as
// 'pick' events. A change of the selected point rebuilds only its layer. Radii and line widths are
// in pixels, so that the points look the same at every zoom.

const SEGMENT_ZOOM = 14;
const HIGHLIGHTING_ALPHA = 50;

function loadScript(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`${url} could not be loaded.`));
    document.head.appendChild(script);
  });
}

let librariesPromise = null;

// MapLibre (the module) and deck.gl (the global `deck`), loaded once per page.
function loadLibraries(maplibreUrl, maplibreStylesheetUrl, deckUrl) {
  if (librariesPromise === null) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = maplibreStylesheetUrl;
    document.head.appendChild(link);
    librariesPromise = Promise.all([import(maplibreUrl), loadScript(deckUrl)]);
  }
  return librariesPromise;
}

function computeBounds(longitudes, latitudes) {
  // A loop, since `Math.min(...array)` exceeds the call stack for large arrays.
  let [west, south, east, north] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let i = 0; i < longitudes.length; i++) {
    west = Math.min(west, longitudes[i]);
    east = Math.max(east, longitudes[i]);
    south = Math.min(south, latitudes[i]);
    north = Math.max(north, latitudes[i]);
  }
  return [west, south, east, north];
}

export default {
  template: '<div class="deck-map" style="position: absolute; inset: 0"></div>',
  props: {
    longitudes: Array,
    latitudes: Array,
    segmentIds: Array,
    maplibreUrl: String,
    maplibreStylesheetUrl: String,
    deckUrl: String,
    basemapStyleUrl: String,
  },
  created() {
    // Not reactive data, since Vue would wrap the arrays in proxies.
    this.outcomes = null;
    this.highlightedPoints = null;
    this.selectedPoint = null;
    this.layers = { segments: null, highlighted: null, selected: null };
  },
  async mounted() {
    let maplibregl;
    try {
      [maplibregl] = await loadLibraries(this.maplibreUrl, this.maplibreStylesheetUrl, this.deckUrl);
    } catch (error) {
      this.$el.textContent = `The map cannot be shown: ${error.message}`;
      return;
    }
    const numPoints = this.longitudes.length;
    this.positions = new Float32Array(numPoints * 2);
    for (let i = 0; i < numPoints; i++) {
      this.positions[2 * i] = this.longitudes[i];
      this.positions[2 * i + 1] = this.latitudes[i];
    }
    this.map = new maplibregl.Map({
      container: this.$el,
      style: this.basemapStyleUrl,
      ...(numPoints > 0
        ? { bounds: computeBounds(this.longitudes, this.latitudes), fitBoundsOptions: { padding: 40 } }
        : { center: [0, 0], zoom: 1 }),
    });
    this.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    this.overlay = new deck.MapboxOverlay({
      interleaved: false,
      // The pointer over a clickable point. The deck.gl canvas does not get pointer events, so
      // the cursor is set on the map canvas.
      onHover: ({ picked }) => { this.map.getCanvas().style.cursor = picked ? "pointer" : ""; },
      getTooltip: ({ index, layer }) => {
        if (index < 0 || !layer || !this.outcomes) return null;
        const point = layer.id === "segments" ? index : layer.props.data[index];
        return `${this.segmentIds[point]}: ${this.outcomeNames[this.outcomes[point]]}`;
      },
    });
    this.map.addControl(this.overlay);
    this.resizeObserver = new ResizeObserver(() => this.map.resize());
    this.resizeObserver.observe(this.$el);
    this.updateOutcomeLayers();
    this.updateSelectedLayer();
  },
  unmounted() {
    this.resizeObserver?.disconnect();
    this.map?.remove();
  },
  methods: {
    // outcomeCodes: one digit per point, the index into `outcomeNames` and `outcomeStyles`.
    // outcomeStyles: [[r, g, b, alpha], radius in pixels] of each outcome.
    set_outcomes(outcomeCodes, outcomeNames, outcomeStyles) {
      this.outcomes = new Uint8Array(outcomeCodes.length);
      for (let i = 0; i < outcomeCodes.length; i++) this.outcomes[i] = outcomeCodes.charCodeAt(i) - 48;
      this.outcomeNames = outcomeNames;
      this.outcomeStyles = outcomeStyles;
      this.updateOutcomeLayers();
    },
    // highlightedPoints: the indices of the highlighted points, or null. Unless it is null, the
    // other points are faded.
    set_highlighted(highlightedPoints) {
      this.highlightedPoints = highlightedPoints;
      this.updateOutcomeLayers();
    },
    // selectedPoint: the index of the selected point, or null.
    set_selected(selectedPoint) {
      this.selectedPoint = selectedPoint;
      this.updateSelectedLayer();
    },
    getPointPosition(point) {
      return [this.positions[2 * point], this.positions[2 * point + 1]];
    },
    setLayers() {
      this.overlay.setProps({ layers: Object.values(this.layers).filter((layer) => layer !== null) });
    },
    // The layers of the points and of the highlighted ones, which depend on the outcomes.
    updateOutcomeLayers() {
      if (!this.overlay || !this.outcomes) return;
      const numPoints = this.positions.length / 2;
      const isHighlighting = this.highlightedPoints !== null;
      const colors = new Uint8Array(numPoints * 4);
      const radii = new Float32Array(numPoints);
      for (let i = 0; i < numPoints; i++) {
        const [color, radius] = this.outcomeStyles[this.outcomes[i]];
        colors.set(color, 4 * i);
        if (isHighlighting) colors[4 * i + 3] = Math.min(color[3], HIGHLIGHTING_ALPHA);
        radii[i] = radius;
      }
      this.layers.segments = new deck.ScatterplotLayer({
        id: "segments",
        data: {
          length: numPoints,
          attributes: {
            getPosition: { value: this.positions, size: 2 },
            getFillColor: { value: colors, size: 4 },
            getRadius: { value: radii, size: 1 },
          },
        },
        radiusUnits: "pixels",
        pickable: true,
        onClick: ({ index }) => this.$emit("pick", index),
      });
      this.layers.highlighted = new deck.ScatterplotLayer({
        id: "highlighted",
        data: this.highlightedPoints ?? [],
        getPosition: (point) => this.getPointPosition(point),
        getFillColor: (point) => [...this.outcomeStyles[this.outcomes[point]][0].slice(0, 3), 255],
        // The data can stay the same while the outcomes change, e.g. with another model B.
        updateTriggers: { getFillColor: this.outcomes },
        getRadius: 6,
        radiusUnits: "pixels",
        stroked: true,
        getLineColor: [255, 255, 255, 255],
        getLineWidth: 1,
        lineWidthUnits: "pixels",
        pickable: true,
        onClick: ({ object }) => this.$emit("pick", object),
      });
      this.setLayers();
    },
    // The ring around the selected point. The map moves to a newly selected point if it is out of
    // view or the map is zoomed out.
    updateSelectedLayer() {
      if (!this.overlay) return;
      const selectedPoint = this.selectedPoint;
      this.layers.selected = new deck.ScatterplotLayer({
        id: "selected",
        data: selectedPoint === null ? [] : [selectedPoint],
        getPosition: (point) => this.getPointPosition(point),
        getLineColor: [20, 20, 20, 255],
        getRadius: 11,
        radiusUnits: "pixels",
        stroked: true,
        filled: false,
        getLineWidth: 2.5,
        lineWidthUnits: "pixels",
      });
      this.setLayers();
      if (selectedPoint !== null && selectedPoint !== this.shownPoint) {
        const position = this.getPointPosition(selectedPoint);
        if (!this.map.getBounds().contains(position) || this.map.getZoom() < SEGMENT_ZOOM - 2)
          this.map.easeTo({ center: position, zoom: Math.max(this.map.getZoom(), SEGMENT_ZOOM) });
      }
      this.shownPoint = selectedPoint;
    },
  },
};
