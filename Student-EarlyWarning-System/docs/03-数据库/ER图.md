# ER图

## 数据实体关系

```mermaid
erDiagram

Student ||--o{ CourseSession : attends

CourseSession ||--o{ FrameRecord : contains

FrameRecord ||--o{ StateRecord : aggregate

StateRecord ||--o{ BehaviorRecord : generate

Student ||--o{ WarningLog : receive

Student {

int student_id PK

string student_no

string student_name

}

CourseSession {

int session_id PK

string course_name

datetime start_time

}

FrameRecord {

int frame_id PK

float ear

float mar

float pitch

float yaw

float roll

}

StateRecord {

int state_id PK

float avg_ear

float avg_mar

float perclos

float attention_score

float fatigue_score

}

BehaviorRecord {

int behavior_id PK

string behavior_type

datetime start_time

datetime end_time

float duration

}

WarningLog {

int warning_id PK

string warning_level

string warning_type

float attention_score

}
```

## 设计说明

数据库采用"原始数据 + 聚合数据"双层存储模式。

* FrameRecord 保存每帧原始视觉特征，用于科研分析与模型训练。
* StateRecord 保存每秒聚合统计结果，用于系统展示和风险分析。
* BehaviorRecord 保存行为事件，而非每帧状态，提高查询效率。
* WarningLog 保存预警信息，为教师端提供风险提示。
