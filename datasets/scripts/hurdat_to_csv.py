import pandas as pd

def parse_hurdat2(filepath):
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    storm_id = None
    storm_name = None
    for line_num, line in enumerate(lines):
        parts = [p.strip() for p in line.split(',') if p.strip() != '']
        if len(parts) == 3:
            # storm header
            storm_id = parts[0]
            storm_name = parts[1]
        elif len(parts) >= 8:
            # storm data line (ensure minimum columns)
            try:
                date = parts[0]
                time = parts[1]
                status = parts[3]
                lat_str = parts[4]
                lon_str = parts[5]
                wind = int(parts[6])
                pressure = int(parts[7]) if parts[7] != '-999' else None

                # convert lat
                lat = float(lat_str[:-1])
                if lat_str[-1] == 'S':
                    lat = -lat

                # convert lon
                lon = float(lon_str[:-1])
                if lon_str[-1] == 'W':
                    lon = -lon

                data.append({
                    'storm_id': storm_id,
                    'storm_name': storm_name,
                    'date': date,
                    'time': time,
                    'status': status,
                    'lat': lat,
                    'lon': lon,
                    'wind': wind,
                    'pressure': pressure
                })
            except Exception as e:
                print(f"⚠️ Skipping malformed line {line_num}: {line.strip()} ({e})")
        else:
            # Skip unexpected line
            print(f"⚠️ Skipping short/unexpected line {line_num}: {line.strip()}")

    df = pd.DataFrame(data)
    return df


# Example use
df = parse_hurdat2('/Users/noellelaw/Desktop/CERALab/TrackCtrl/datasets/scripts/hurdat2-1851-2024-040425.txt')
df.to_csv('hurdat2_north_atlantic.csv', index=False)
print("Converted to CSV: hurdat2_north_atlantic.csv")
print(df.head())