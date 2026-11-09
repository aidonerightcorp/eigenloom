import { atom, cn, host, ListRow, SegmentedControl, Tip, ToggleRow, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

const PAPER_INK_STYLE = {
  backgroundColor: '#f5f2ea',
  color: '#141413',
  colorScheme: 'only light',
  border: '1px solid #cfc8b8',
  borderRadius: '8px',
  padding: '14px',
  fontFamily: 'ui-sans-serif, system-ui, sans-serif'
}

const CARD_STYLE = {
  backgroundColor: '#faf8f3',
  color: '#141413',
  border: '1px solid #cfc8b8',
  borderRadius: '6px',
  padding: '10px',
  marginTop: '8px'
}

const DEFAULTS = {
  showStatusChip: true,
  mrlSlice: '64'
}

const INITIAL_TELEMETRY = {
  version: '1.0.0',
  engine_count: 20,
  sub_engine_count: 72,
  hypergrid_total_dimensions: 100,
  matryoshka_spaces: ['task2vec', 'config2vec', 'weight2vec', 'trace2vec'],
  matryoshka_prefix_dims: [16, 32, 64, 128],
  persisted_tasks: 0,
  persisted_verifications: 0,
  persisted_attractors: 0
}

function EigenLoomStatusChip({ prefs, telemetry }) {
  const gateway = useValue(host.state.gateway)
  const { showStatusChip, mrlSlice } = useValue(prefs)
  const snap = useValue(telemetry)

  if (!showStatusChip) {
    return null
  }

  return jsx(Tip, {
    label: `EigenLoom v${snap.version} • ${snap.engine_count} engines / ${snap.sub_engine_count} sub-engines • gateway: ${gateway}`,
    children: jsxs('span', {
      className: cn('inline-flex h-full items-center gap-1.5 px-2 text-[0.6875rem]'),
      style: {
        backgroundColor: '#f5f2ea',
        color: '#141413',
        colorScheme: 'only light',
        borderLeft: '1px solid #cfc8b8'
      },
      children: [
        jsx('span', { 'aria-hidden': true, children: '◈' }),
        jsx('span', { children: `EigenLoom ${snap.engine_count}E · ${mrlSlice}d` })
      ]
    })
  })
}

function EigenLoomInspectorPane({ prefs, telemetry, onRefresh }) {
  const { mrlSlice } = useValue(prefs)
  const snap = useValue(telemetry)

  return jsxs('div', {
    style: PAPER_INK_STYLE,
    children: [
      jsxs('div', {
        style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
        children: [
          jsxs('div', {
            children: [
              jsx('div', {
                style: { fontSize: '15px', fontWeight: 700, color: '#141413' },
                children: 'EigenLoom Cybernetic Engine Manifold'
              }),
              jsx('div', {
                style: { fontSize: '12px', color: '#4a4843', marginTop: '2px' },
                children: `v${snap.version} • 5 Strata • ${snap.engine_count} Engines • ${snap.sub_engine_count} Sub-Engines • ${snap.hypergrid_total_dimensions}D Fluid HyperGrid`
              })
            ]
          }),
          jsx('button', {
            type: 'button',
            onClick: onRefresh,
            style: {
              backgroundColor: '#ebe6da',
              color: '#141413',
              border: '1px solid #cfc8b8',
              borderRadius: '4px',
              padding: '4px 10px',
              fontSize: '12px',
              cursor: 'pointer'
            },
            children: 'Sync WAL'
          })
        ]
      }),
      jsxs('div', {
        style: CARD_STYLE,
        children: [
          jsx('div', {
            style: { fontSize: '12px', fontWeight: 600, color: '#2f5d50' },
            children: 'Unified 4-Space Matryoshka Geometry'
          }),
          jsx('div', {
            style: { fontSize: '12px', color: '#141413', marginTop: '4px' },
            children: `Spaces: ${(snap.matryoshka_spaces || []).join(' · ')} | Active Slice: ${mrlSlice}d of [${(snap.matryoshka_prefix_dims || []).join(', ')}]`
          })
        ]
      }),
      jsxs('div', {
        style: CARD_STYLE,
        children: [
          jsx('div', {
            style: { fontSize: '12px', fontWeight: 600, color: '#354f6b' },
            children: 'Profile-Scoped SQLite WAL State (Read-Only)'
          }),
          jsx('div', {
            style: { fontSize: '12px', color: '#141413', marginTop: '4px' },
            children: `Fenced Tasks: ${snap.persisted_tasks} • Maker≠Checker Records: ${snap.persisted_verifications} • Hopfield Attractors: ${snap.persisted_attractors}`
          })
        ]
      })
    ]
  })
}

export default {
  id: 'eigenloom',
  name: 'EigenLoom',
  description: 'Cybernetic 20-engine manifold, fluid 100D Matryoshka HyperGrid, and 7-target operator stack compiler.',
  register(ctx) {
    const prefs = atom({ ...DEFAULTS, ...ctx.storage.get('prefs', {}) })
    const telemetry = atom(INITIAL_TELEMETRY)

    const updatePrefs = patch => {
      prefs.set({ ...prefs.get(), ...patch })
      ctx.storage.set('prefs', prefs.get())
    }

    const refreshTelemetry = () => {
      if (typeof ctx.rest !== 'function') {
        return
      }
      Promise.resolve(ctx.rest('/status'))
        .then(data => {
          if (data && typeof data === 'object') {
            telemetry.set({ ...telemetry.get(), ...data })
          }
        })
        .catch(() => {})
    }

    refreshTelemetry()

    ctx.register({
      id: 'eigenloom-chip',
      area: 'statusBar.right',
      order: 105,
      render: () => jsx(EigenLoomStatusChip, { prefs, telemetry })
    })

    ctx.register({
      id: 'eigenloom-pane',
      area: 'panes',
      title: 'EigenLoom Loom',
      icon: 'cpu',
      render: () => jsx(EigenLoomInspectorPane, { prefs, telemetry, onRefresh: refreshTelemetry })
    })

    ctx.registerSettingsPage?.({
      id: 'settings',
      title: 'EigenLoom',
      icon: 'cpu',
      render: function GeneralSettings() {
        const { showStatusChip, mrlSlice } = useValue(prefs)
        return jsxs('div', {
          style: { colorScheme: 'only light' },
          children: [
            jsx(ToggleRow, {
              checked: showStatusChip,
              description: 'Display live 20-engine and Matryoshka slice indicator in the status bar.',
              label: 'Show EigenLoom status chip',
              onChange: enabled => updatePrefs({ showStatusChip: enabled })
            }),
            jsx(ListRow, {
              action: jsx(SegmentedControl, {
                onChange: value => updatePrefs({ mrlSlice: value }),
                options: [
                  { id: '16', label: '16d' },
                  { id: '32', label: '32d' },
                  { id: '64', label: '64d' },
                  { id: '128', label: '128d' }
                ],
                value: mrlSlice
              }),
              description: 'Select default nested Matryoshka prefix slice for inspection.',
              title: 'Matryoshka slice dimension'
            })
          ]
        })
      }
    })
  }
}
