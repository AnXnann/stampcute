import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ สร้างข้อมูลตามเกณฑ์สากล
data = {
    'Glucose': [90, 100, 110, 130, 80, 70, 140, 125, 160, 180, 50, 200, 150, 170],
    'Cholesterol': [180, 200, 210, 250, 160, 190, 300, 220, 230, 240, 150, 270, 200, 260],
    'BloodPressure': [110, 120, 125, 135, 105, 115, 140, 128, 130, 145, 100, 150, 140, 155],
    'Risk': ['ปลอดภัย', 'เสี่ยง', 'เสี่ยง', 'อันตราย', 'ปลอดภัย', 'ปลอดภัย',
             'อันตราย', 'เสี่ยง', 'อันตราย', 'อันตราย', 'อันตราย', 'อันตราย',
             'อันตราย', 'อันตราย']
}

df = pd.DataFrame(data)

# 2️⃣ แยก features และ labels
X = df[['Glucose', 'Cholesterol', 'BloodPressure']]
y = df['Risk']

# 3️⃣ แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ สร้างโมเดล Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 5️⃣ ประเมินความแม่นยำ
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# 6️⃣ ทำนายค่าที่ user ใส่
new_data = [[2000, 2000, 2000]]  # ใส่ค่า Glucose, Cholesterol, BloodPressure
risk_pred = model.predict(new_data)

print(f"ค่าที่ตรวจ: Glucose={new_data[0][0]}, Cholesterol={new_data[0][1]}, BP={new_data[0][2]}")
print(f"ระดับความเสี่ยง (AI ทำนาย): {risk_pred[0]}")
