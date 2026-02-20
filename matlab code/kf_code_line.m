%% 参数设置
clear; clc; close all;
dt = 0.1;               % 步长 (秒)
total_time = 10;        % 一圈总时间 (秒)
steps = total_time / dt; % 步数
t = linspace(0, total_time, steps+1);  

%% 实际轨迹（直线）
x_true = 2 * t;  % 直线路径：x方向匀速运动
y_true = 3 * t;  % 直线路径：y方向匀速运动
% 可以根据需要调整直线方向，例如：
% x_true = 5 * t; y_true = 0 * t;  % 水平直线
% x_true = 0 * t; y_true = 5 * t;  % 垂直直线

% 计算真实速度和加速度（用于生成传感器测量值）
vx_true = gradient(x_true) / dt;
vy_true = gradient(y_true) / dt;
ax_true = gradient(vx_true) / dt;
ay_true = gradient(vy_true) / dt;

%% 传感器测量

% GPS 测量值（添加噪声）
sigma_gps = 0.5; % GPS噪声标准差
x_meas = x_true + sigma_gps * randn(size(x_true));
y_meas = y_true + sigma_gps * randn(size(y_true));

% 速度传感器测量值（添加噪声）
sigma_vel = 0.3; % 速度传感器噪声标准差
vx_meas = vx_true + sigma_vel * randn(size(vx_true));
vy_meas = vy_true + sigma_vel * randn(size(vy_true));

% 计算全局视角范围（添加边距）
x_min = min(x_true) - 2;
x_max = max(x_true) + 2;
y_min = min(y_true) - 2;
y_max = max(y_true) + 2;

%% 卡尔曼滤波初始化
n = 6;  % 状态维度 [x; y; vx; vy; ax; ay]
m = 4;  % 观测维度 (GPS的x,y + 加速度计的vx,vy)

% 初始状态估计（从真实起点开始）
x_est = [x_true(1); y_true(1); vx_true(1); vy_true(1); ax_true(1); ay_true(1)];
%可以全部改为别的数值，验证卡尔曼滤波的收敛性，但是要适当调大下面P的方差

% 初始误差协方差矩阵（因为初始状态已知，方差设小）
P = diag([0.01, 0.01, 0.1, 0.1, 0.5, 0.5]);

% 状态转移矩阵（常加速度模型）
% x_k+1 = x_k + vx*dt + 0.5*ax*dt^2
% vx_k+1 = vx_k + ax*dt
% ax_k+1 = ax_k (假设加速度不变)
F = [1, 0, dt, 0,  0.5*dt^2, 0;
     0, 1, 0,  dt, 0,        0.5*dt^2;
     0, 0, 1,  0,  dt,       0;
     0, 0, 0,  1,  0,        dt;
     0, 0, 0,  0,  1,        0;
     0, 0, 0,  0,  0,        1];

% 过程噪声协方差矩阵（调小，因为模型更准确）
q_pos = 0.001;   % 位置过程噪声
q_vel = 0.001;   % 速度过程噪声
q_acc = 0.005;    % 加速度过程噪声
Q = diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc]);

% 观测矩阵（观测位置和速度）
H = [1, 0, 0, 0, 0, 0;   % 观测x
     0, 1, 0, 0, 0, 0;   % 观测y
     0, 0, 1, 0, 0, 0;   % 观测vx  
     0, 0, 0, 1, 0, 0];  % 观测vy  

% 测量噪声协方差矩阵
R = diag([(sigma_gps)^2, (sigma_gps)^2, (sigma_vel)^2, (sigma_vel)^2]); 

% 存储卡尔曼估计结果
x_kalman = zeros(2, steps+1);
x_kalman(:, 1) = x_est(1:2);

fprintf('卡尔曼滤波初始化完成（6维状态：位置+速度+加速度）\n');

%% 卡尔曼滤波循环 
for k = 1:steps
   
    % 预测步
    x_pred = F * x_est;
    P_pred = F * P * F' + Q;
    
     % 当前测量值（GPS位置 + 速度计）
    y_k = [x_meas(k+1); y_meas(k+1); vx_meas(k+1); vy_meas(k+1)]; 
    
    % 更新步
    K = P_pred * H' / (H * P_pred * H' + R);
    x_est = x_pred + K * (y_k - H * x_pred);
    P = (eye(n) - K * H) * P_pred;
    
    % 存储估计位置
    x_kalman(:, k+1) = x_est(1:2);
end

%% 逐步绘图 
figure;
hold on;
h_true = plot(nan, nan, 'k-', 'LineWidth', 2);
h_gps = plot(nan, nan, 'b.', 'MarkerSize', 8);  % 添加GPS点句柄
h_kalman = plot(nan, nan, 'g-', 'LineWidth', 2);

xlabel('X (m)');
ylabel('Y (m)');
title('卡尔曼滤波预测 vs 实际路径（带速度传感器）');
axis equal;
xlim([x_min, x_max]);
ylim([y_min, y_max]);
grid on;
legend([h_true, h_gps, h_kalman], {'实际路径', 'GPS测量值', '卡尔曼预测（GPS+速度计）'});  % 更新图例

% 逐步绘制
for i = 1:steps+1
    set(h_true, 'XData', x_true(1:i), 'YData', y_true(1:i));
    set(h_gps, 'XData', x_meas(1:i), 'YData', y_meas(1:i));  % 添加GPS点更新
    set(h_kalman, 'XData', x_kalman(1, 1:i), 'YData', x_kalman(2, 1:i));
    
    drawnow;
    pause(dt);
end
%% 误差对比分析
% 计算每个点的误差
gps_errors = sqrt((x_meas - x_true).^2 + (y_meas - y_true).^2);
kalman_errors = sqrt((x_kalman(1,:) - x_true).^2 + (x_kalman(2,:) - y_true).^2);

% 计算统计指标
gps_mean_error = mean(gps_errors);
gps_max_error = max(gps_errors);
gps_std_error = std(gps_errors);

kalman_mean_error = mean(kalman_errors);
kalman_max_error = max(kalman_errors);
kalman_std_error = std(kalman_errors);

% 计算提升百分比
improvement_mean = (gps_mean_error - kalman_mean_error) / gps_mean_error * 100;
improvement_max = (gps_max_error - kalman_max_error) / gps_max_error * 100;

% 显示结果
fprintf('\n========== 误差对比分析 ==========\n');
fprintf('%-25s %15s %15s %15s\n', '指标', 'GPS', '卡尔曼滤波', '提升百分比');
fprintf('%-25s %15.4f %15.4f %14.1f%%\n', '平均误差 (m)', gps_mean_error, kalman_mean_error, improvement_mean);
fprintf('%-25s %15.4f %15.4f %14.1f%%\n', '最大误差 (m)', gps_max_error, kalman_max_error, improvement_max);
fprintf('%-25s %15.4f %15.4f\n', '标准差 (m)', gps_std_error, kalman_std_error);
fprintf('====================================\n');
fprintf('绘图完成！\n');