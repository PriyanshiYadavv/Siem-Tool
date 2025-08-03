# Complete SIEM Tool - All modules in one file for easy testing
# Save this as siem_tool.py and run with: streamlit run siem_tool.py

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="SIEM Log Analysis Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# 1. LOG PARSER MODULE
# ================================

class LogParser:
    def __init__(self):
        # Enhanced regex pattern for better log parsing
        self.log_pattern = re.compile(
            r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<url>\S+) HTTP/\d\.\d" (?P<status>\d{3}) (?P<size>\S+) "(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)"'
        )
    
    def parse_apache_time(self, timestamp):
        """Convert Apache timestamp to UTC datetime object"""
        try:
            dt = datetime.strptime(timestamp.split()[0], "%d/%b/%Y:%H:%M:%S")
            return dt
        except:
            return datetime.now()
    
    def parse_log_file(self, file_path):
        """Parse log file and return DataFrame"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        match = self.log_pattern.search(line)
                        if match:
                            info = match.groupdict()
                            info['timestamp'] = self.parse_apache_time(info['timestamp'])
                            info['status'] = int(info['status'])
                            data.append(info)
                    except Exception as e:
                        continue
        except FileNotFoundError:
            st.error(f"File {file_path} not found")
            return pd.DataFrame()
        
        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            return df
        else:
            return pd.DataFrame()
    
    def create_sample_data(self, num_records=1000):
        """Create sample log data for testing"""
        import random
        from datetime import datetime, timedelta
        
        # Sample IPs with some suspicious ones
        normal_ips = ['192.168.1.10', '192.168.1.20', '10.0.0.5', '172.16.0.100']
        suspicious_ips = ['45.33.32.156', '185.220.101.182', '198.51.100.1']
        all_ips = normal_ips + suspicious_ips
        
        # Sample URLs
        urls = ['/index.html', '/login.php', '/admin.php', '/wp-admin/', '/dashboard', 
                '/api/users', '/config.php', '/backup.sql', '/robots.txt', '/favicon.ico']
        
        # Sample User Agents
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'python-requests/2.25.1',
            'curl/7.68.0',
            'Googlebot/2.1'
        ]
        
        data = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(num_records):
            # Create some attack patterns
            if random.random() < 0.1:  # 10% suspicious activity
                ip = random.choice(suspicious_ips)
                url = random.choice(['/login.php', '/admin.php', '/wp-admin/'])
                status = random.choice([403, 404, 500])
                method = 'POST'
            else:
                ip = random.choice(all_ips)
                url = random.choice(urls)
                status = random.choice([200, 200, 200, 200, 301, 302, 404, 403, 500])
                method = random.choice(['GET', 'POST', 'PUT', 'DELETE'])
            
            timestamp = base_time + timedelta(seconds=random.randint(0, 86400))
            
            data.append({
                'ip': ip,
                'timestamp': timestamp,
                'method': method,
                'url': url,
                'status': status,
                'size': random.randint(200, 50000),
                'referrer': random.choice(['-', 'https://google.com', 'https://example.com']),
                'user_agent': random.choice(user_agents)
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')

# ================================
# 2. FEATURE ENGINEERING MODULE
# ================================

class FeatureEngineer:
    def __init__(self):
        pass
    
    def is_business_hours(self, timestamp):
        """Check if timestamp is within business hours (9 AM - 6 PM, Monday-Friday)"""
        return (timestamp.weekday() < 5 and 9 <= timestamp.hour <= 18)
    
    def engineer_features(self, df):
        """Create features from log data"""
        if df.empty:
            return pd.DataFrame()
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.date
        df['is_business_hours'] = df['timestamp'].apply(self.is_business_hours)
        
        # Group by IP and calculate features
        features = df.groupby('ip').agg({
            'url': ['count', 'nunique'],
            'user_agent': 'nunique',
            'referrer': 'nunique',
            'status': [
                lambda x: sum((x >= 200) & (x < 300)),  # 2xx
                lambda x: sum((x >= 300) & (x < 400)),  # 3xx
                lambda x: sum((x >= 400) & (x < 500)),  # 4xx
                lambda x: sum((x >= 500) & (x < 600)),  # 5xx
                lambda x: sum(x == 403),  # 403 errors
                lambda x: sum(x == 404),  # 404 errors
            ],
            'timestamp': lambda x: x.sort_values().diff().dropna().mean().total_seconds() if len(x) > 1 else 0,
            'hour': lambda x: x.mode()[0] if not x.mode().empty else -1,
            'is_business_hours': lambda x: sum(~x),  # After hours requests
        }).reset_index()
        
        # Flatten column names
        features.columns = [
            'ip', 'request_count', 'unique_urls', 'unique_user_agents', 
            'referrer_count', 'count_2xx', 'count_3xx', 'count_4xx', 
            'count_5xx', 'burst_403', 'count_404', 'avg_time_between_requests',
            'time_of_day_mode', 'after_hours_requests'
        ]
        
        return features
    
    def detect_scan_patterns(self, df):
        """Detect URL scanning patterns"""
        scan_alerts = []
        for ip, group in df.groupby('ip'):
            urls = group['url'].tolist()
            similar_pairs = 0
            for i in range(len(urls)):
                for j in range(i+1, min(len(urls), i+50)):  # Limit comparison for performance
                    similarity = SequenceMatcher(None, urls[i], urls[j]).ratio()
                    if similarity > 0.8:
                        similar_pairs += 1
            
            if similar_pairs > 20:
                scan_alerts.append((ip, 'Potential Scan Pattern'))
        
        return pd.DataFrame(scan_alerts, columns=['ip', 'alert'])

# ================================
# 3. ANOMALY DETECTION MODULE
# ================================

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = None
        self.random_forest = None
        self.scaler = StandardScaler()
    
    def detect_anomalies_isolation_forest(self, df_features):
        """Detect anomalies using Isolation Forest"""
        if df_features.empty:
            return df_features
        
        # Prepare features for ML
        feature_cols = [col for col in df_features.columns if col != 'ip']
        features = df_features[feature_cols].fillna(0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply Isolation Forest
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
        # First fit the model, then get predictions
        self.isolation_forest.fit(features_scaled)
        df_features['anomaly_score'] = self.isolation_forest.decision_function(features_scaled)
        df_features['anomaly'] = self.isolation_forest.predict(features_scaled)
        df_features['anomaly'] = df_features['anomaly'].map({1: 0, -1: 1})  # 1 = anomaly, 0 = normal
        
        return df_features
    
    def train_supervised_model(self, df_features, labels):
        """Train supervised model if labeled data is available"""
        if df_features.empty:
            return None
        
        feature_cols = [col for col in df_features.columns if col not in ['ip', 'anomaly', 'anomaly_score']]
        X = df_features[feature_cols].fillna(0)
        y = labels
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.random_forest.fit(X_train, y_train)
        
        y_pred = self.random_forest.predict(X_test)
        return classification_report(y_test, y_pred)

# ================================
# 4. RULE ENGINE MODULE
# ================================

class RuleEngine:
    def __init__(self):
        pass
    
    def apply_siem_rules(self, df):
        """Apply SIEM rules to detect threats"""
        alerts = []
        if df.empty:
            return pd.DataFrame(columns=['ip', 'alert', 'timestamp'])
        
        df_sorted = df.sort_values(by='timestamp')
        
        for ip, group in df_sorted.groupby('ip'):
            group = group.reset_index(drop=True)
            
            # Rule 1: Multiple failed logins (403) in 1 minute
            fails = group[group['status'] == 403].reset_index(drop=True)
            if len(fails) >= 5:
                for i in range(len(fails) - 4):
                    time_diff = (fails.iloc[i+4]['timestamp'] - fails.iloc[i]['timestamp']).total_seconds()
                    if time_diff <= 60:
                        alerts.append((ip, 'Multiple Failed Logins', fails.iloc[i]['timestamp']))
                        break
            
            # Rule 2: Success after failures
            statuses = group['status'].tolist()
            if 403 in statuses and 200 in statuses:
                try:
                    if statuses.index(200) > statuses.index(403):
                        alerts.append((ip, 'Success after Failures', group.iloc[0]['timestamp']))
                except ValueError:
                    pass
            
            # Rule 3: Request flood (>50 requests in 1 minute)
            if len(group) >= 50:
                for i in range(len(group) - 49):
                    time_diff = (group.iloc[i+49]['timestamp'] - group.iloc[i]['timestamp']).total_seconds()
                    if time_diff <= 60:
                        alerts.append((ip, 'Request Flood', group.iloc[i]['timestamp']))
                        break
            
            # Rule 4: 404 error spike
            errors_404 = group[group['status'] == 404]
            if len(errors_404) >= 10:
                alerts.append((ip, '404 Error Spike', group.iloc[0]['timestamp']))
            
            # Rule 5: After hours activity
            after_hours = group[~group['timestamp'].apply(lambda x: (x.weekday() < 5 and 9 <= x.hour <= 18))]
            if len(after_hours) > 20:
                alerts.append((ip, 'After Hours Activity', after_hours.iloc[0]['timestamp']))
        
        return pd.DataFrame(alerts, columns=['ip', 'alert', 'timestamp'])

# ================================
# 5. VISUALIZATION MODULE
# ================================

class Visualizer:
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_request_distribution(self, df_features):
        """Plot request count per IP"""
        if df_features.empty:
            st.write("No data to display")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        top_ips = df_features.nlargest(10, 'request_count')
        
        sns.barplot(data=top_ips, x='request_count', y='ip', ax=ax)
        ax.set_title('Top 10 IPs by Request Count')
        ax.set_xlabel('Number of Requests')
        ax.set_ylabel('IP Address')
        
        st.pyplot(fig)
    
    def plot_status_codes(self, df):
        """Plot status code distribution"""
        if df.empty:
            st.write("No data to display")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        status_counts = df['status'].value_counts().sort_index()
        
        colors = ['green' if x < 400 else 'orange' if x < 500 else 'red' for x in status_counts.index]
        bars = ax.bar(status_counts.index.astype(str), status_counts.values, color=colors)
        
        ax.set_title('HTTP Status Code Distribution')
        ax.set_xlabel('Status Code')
        ax.set_ylabel('Count')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    def plot_anomalies(self, df_anomalies):
        """Plot anomaly scores"""
        if df_anomalies.empty or 'anomaly_score' not in df_anomalies.columns:
            st.write("No anomaly data to display")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by anomaly score and take top 15
        top_anomalies = df_anomalies.nsmallest(15, 'anomaly_score')
        
        colors = ['red' if x == 1 else 'green' for x in top_anomalies['anomaly']]
        bars = ax.barh(range(len(top_anomalies)), top_anomalies['anomaly_score'], color=colors)
        
        ax.set_yticks(range(len(top_anomalies)))
        ax.set_yticklabels(top_anomalies['ip'])
        ax.set_xlabel('Anomaly Score (lower = more anomalous)')
        ax.set_title('Top 15 IPs by Anomaly Score')
        
        st.pyplot(fig)
    
    def plot_time_series(self, df):
        """Plot requests over time"""
        if df.empty:
            st.write("No data to display")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Resample to hourly counts
        hourly_counts = df.set_index('timestamp').resample('H').size()
        
        ax.plot(hourly_counts.index, hourly_counts.values, marker='o', linewidth=2)
        ax.set_title('Requests Over Time (Hourly)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Requests')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# ================================
# 6. MAIN DASHBOARD
# ================================

def main_dashboard():
    st.title("🔒 SIEM Log Analysis Dashboard")
    st.markdown("---")
    
    # Initialize components
    parser = LogParser()
    feature_engineer = FeatureEngineer()
    anomaly_detector = AnomalyDetector()
    rule_engine = RuleEngine()
    visualizer = Visualizer()
    
    # Sidebar
    st.sidebar.header("📊 Data Source")
    data_source = st.sidebar.selectbox(
        "Choose data source:",
        ["Sample Data", "Upload Log File"]
    )
    
    # Load data
    if data_source == "Sample Data":
        st.sidebar.info("Using generated sample data for demonstration")
        num_records = st.sidebar.slider("Number of sample records", 100, 5000, 1000)
        
        with st.spinner("Generating sample data..."):
            df_logs = parser.create_sample_data(num_records)
    
    else:
        uploaded_file = st.sidebar.file_uploader("Upload log file", type=['log', 'txt'])
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with open("temp_log.txt", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Parsing log file..."):
                df_logs = parser.parse_log_file("temp_log.txt")
        else:
            st.warning("Please upload a log file or use sample data")
            return
    
    if df_logs.empty:
        st.error("No valid log data found!")
        return
    
    # Process data
    with st.spinner("Processing data..."):
        df_features = feature_engineer.engineer_features(df_logs)
        df_anomalies = anomaly_detector.detect_anomalies_isolation_forest(df_features)
        alerts = rule_engine.apply_siem_rules(df_logs)
        scan_alerts = feature_engineer.detect_scan_patterns(df_logs)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Requests", len(df_logs))
    
    with col2:
        st.metric("🌐 Unique IPs", df_logs['ip'].nunique())
    
    with col3:
        anomaly_count = df_anomalies['anomaly'].sum() if not df_anomalies.empty else 0
        st.metric("🚨 Anomalies Detected", anomaly_count)
    
    with col4:
        error_rate = (df_logs['status'] >= 400).mean()
        st.metric("💥 Error Rate", f"{error_rate:.2%}")
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🚨 Alerts", "🔍 Anomalies", "📊 Analytics", "📋 Raw Data"])
    
    with tab1:
        st.subheader("📈 Request Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top IPs by Request Count")
            visualizer.plot_request_distribution(df_features)
        
        with col2:
            st.subheader("Status Code Distribution")
            visualizer.plot_status_codes(df_logs)
        
        st.subheader("Requests Over Time")
        visualizer.plot_time_series(df_logs)
    
    with tab2:
        st.subheader("🚨 Security Alerts")
        
        if not alerts.empty:
            st.dataframe(alerts, use_container_width=True)
            
            # Alert summary
            alert_summary = alerts['alert'].value_counts()
            st.subheader("Alert Summary")
            st.bar_chart(alert_summary)
        else:
            st.success("No security alerts detected!")
        
        if not scan_alerts.empty:
            st.subheader("Scan Pattern Alerts")
            st.dataframe(scan_alerts, use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Anomaly Detection Results")
        
        if not df_anomalies.empty:
            visualizer.plot_anomalies(df_anomalies)
            
            st.subheader("Anomalous IPs")
            anomalous_ips = df_anomalies[df_anomalies['anomaly'] == 1]
            if not anomalous_ips.empty:
                st.dataframe(anomalous_ips[['ip', 'anomaly_score', 'request_count', 'count_4xx', 'count_5xx']], 
                           use_container_width=True)
            else:
                st.info("No anomalous IPs detected")
    
    with tab4:
        st.subheader("📊 Detailed Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Request Methods")
            if 'method' in df_logs.columns:
                method_counts = df_logs['method'].value_counts()
                st.bar_chart(method_counts)
        
        with col2:
            st.subheader("Hourly Activity Pattern")
            hourly_activity = df_logs.groupby(df_logs['timestamp'].dt.hour).size()
            st.line_chart(hourly_activity)
        
        st.subheader("Feature Analysis")
        if not df_features.empty:
            st.dataframe(df_features, use_container_width=True)
    
    with tab5:
        st.subheader("📋 Raw Log Data")
        st.dataframe(df_logs, use_container_width=True)
        
        # Download button
        csv = df_logs.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='log_data.csv',
            mime='text/csv',
        )

# ================================
# 7. RUN THE APPLICATION
# ================================

if __name__ == "__main__":
    main_dashboard()