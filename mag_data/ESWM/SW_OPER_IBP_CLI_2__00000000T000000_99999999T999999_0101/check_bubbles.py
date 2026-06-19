from hapiclient import hapi
import pandas as pd
server='https://vires.services/hapi'
start='2026-06-12T23:59:59Z'
stop='2026-06-13T23:59:59Z'
datasets={'Swarm A': 'SW_OPER_IBIATMS_2F', 'Swarm B': 'SW_OPER_IBIBTMS_2F', 'Swarm C': 'SW_OPER_IBICTMS_2F'}
for sat, ds in datasets.items():
    try:
        data, _ = hapi(server, ds, 'Bubble_Index', start, stop)
        df = pd.DataFrame(data)
        if 'Bubble_Index' in df.columns:
            print(sat, 'Total:', len(df), 'Bubbles:', len(df[df['Bubble_Index']==1]))
        else:
            print(sat, 'No Bubble_Index column')
    except Exception as e:
        print(sat, 'Error:', e)
