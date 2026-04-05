function planner_log_app
% PLANNER_LOG_APP
% MATLAB GUI for analyzing planner CSV logs.
%
% Usage:
%   planner_log_app
%
% Features:
% - Select one CSV file
% - Analyze and show overall summary
% - Detect CHANGE maneuvers
% - Plot metrics in tabs
% - Export summary CSV files

    app = struct();
    app.CurrentFile = '';
    app.LastResult = [];
    app.LastTable = table();

    app.UIFigure = uifigure('Name', 'Planner Log Analyzer', ...
        'Position', [80 60 1400 820], ...
        'Color', [0.98 0.98 0.98]);

    gl = uigridlayout(app.UIFigure, [3 1]);
    gl.RowHeight = {95, 230, '1x'};
    gl.ColumnWidth = {'1x'};
    gl.Padding = [10 10 10 10];
    gl.RowSpacing = 10;

    %% Top control panel
    topPanel = uipanel(gl, 'Title', 'Điều khiển');
    topGrid = uigridlayout(topPanel, [3 9]);
    topGrid.RowHeight = {28, 28, 28};
    topGrid.ColumnWidth = {110, '1x', 110, 110, 110, 110, 110, 110, 110};
    topGrid.Padding = [8 8 8 8];

    uilabel(topGrid, 'Text', 'CSV file:', 'FontWeight', 'bold');
    app.FileEdit = uieditfield(topGrid, 'text', 'Editable', 'off');
    app.FileEdit.Layout.Column = [2 9];

    app.BrowseButton = uibutton(topGrid, 'Text', 'Chọn file CSV', ...
        'ButtonPushedFcn', @onBrowse);
    app.BrowseButton.Layout.Row = 2;
    app.BrowseButton.Layout.Column = 1;
    %% Phân tích only Change 
    app.ChangeOnlyButton = uibutton(topGrid, 'Text', 'Phân tích CHANGE', ...
    'ButtonPushedFcn', @onAnalyzeChangeOnly, 'Enable', 'off');
    app.ChangeOnlyButton.Layout.Row = 2;
    app.ChangeOnlyButton.Layout.Column = 3;

    app.AnalyzeButton = uibutton(topGrid, 'Text', 'Phân tích', ...
        'ButtonPushedFcn', @onAnalyze, 'Enable', 'off');
    app.AnalyzeButton.Layout.Row = 2;
    app.AnalyzeButton.Layout.Column = 4;

    app.ExportButton = uibutton(topGrid, 'Text', 'Xuất summary', ...
        'ButtonPushedFcn', @onExport, 'Enable', 'off');
    app.ExportButton.Layout.Row = 2;
    app.ExportButton.Layout.Column = 5;

    app.ClearButton = uibutton(topGrid, 'Text', 'Xóa màn hình', ...
        'ButtonPushedFcn', @onClear);
    app.ClearButton.Layout.Row = 2;
    app.ClearButton.Layout.Column = 6;

    app.VisibleCheck = uicheckbox(topGrid, 'Text', 'Hiện figure riêng', 'Value', false);
    app.VisibleCheck.Layout.Row = 2;
    app.VisibleCheck.Layout.Column = 7;

    app.SaveFigCheck = uicheckbox(topGrid, 'Text', 'Lưu PNG', 'Value', true);
    app.SaveFigCheck.Layout.Row = 2;
    app.SaveFigCheck.Layout.Column = 8;

    app.ExportCheck = uicheckbox(topGrid, 'Text', 'Xuất CSV', 'Value', true);
    app.ExportCheck.Layout.Row = 2;
    app.ExportCheck.Layout.Column = 9;

    uilabel(topGrid, 'Text', 'Thư mục output:', 'FontWeight', 'bold');
    app.OutputEdit = uieditfield(topGrid, 'text', 'Value', pwd);
    app.OutputEdit.Layout.Row = 3;
    app.OutputEdit.Layout.Column = [2 7];

    app.OutputBrowseButton = uibutton(topGrid, 'Text', 'Chọn output', ...
        'ButtonPushedFcn', @onBrowseOutput);
    app.OutputBrowseButton.Layout.Row = 3;
    app.OutputBrowseButton.Layout.Column = 8;

    app.StatusLabel = uilabel(topGrid, 'Text', 'Sẵn sàng.', 'FontColor', [0.1 0.3 0.1]);
    app.StatusLabel.Layout.Row = 3;
    app.StatusLabel.Layout.Column = 9;

    %% Middle panel: tables
    midGrid = uigridlayout(gl, [1 2]);
    midGrid.ColumnWidth = {'1x', '1.4x'};
    midGrid.RowHeight = {'1x'};
    midGrid.ColumnSpacing = 10;

    summaryPanel = uipanel(midGrid, 'Title', 'Overall Summary');
    sumGrid = uigridlayout(summaryPanel, [1 1]);
    app.SummaryTable = uitable(sumGrid, 'Data', table(), 'ColumnSortable', true);

    maneuverPanel = uipanel(midGrid, 'Title', 'Maneuver Summary (CHANGE)');
    manGrid = uigridlayout(maneuverPanel, [1 1]);
    app.ManeuverTable = uitable(manGrid, 'Data', table(), 'ColumnSortable', true);

    %% Bottom panel: plots and text
    bottomGrid = uigridlayout(gl, [1 2]);
    bottomGrid.ColumnWidth = {'3.2x', '1.1x'};

    plotPanel = uipanel(bottomGrid, 'Title', 'Biểu đồ');
    plotGrid = uigridlayout(plotPanel, [1 1]);
    app.TabGroup = uitabgroup(plotGrid);

    app.TabPlanner = uitab(app.TabGroup, 'Title', 'Planner/Safety');
    plannerGrid = uigridlayout(app.TabPlanner, [3 2]);
    app.AxState = uiaxes(plannerGrid); title(app.AxState, 'State timeline');
    app.AxObs = uiaxes(plannerGrid); title(app.AxObs, 'Obstacle distance');
    app.AxMinDist = uiaxes(plannerGrid); title(app.AxMinDist, 'Min planner distance');
    app.AxTTC = uiaxes(plannerGrid); title(app.AxTTC, 'Min TTC');
    app.AxCost = uiaxes(plannerGrid); title(app.AxCost, 'Cost');
    app.AxLane = uiaxes(plannerGrid); title(app.AxLane, 'Lane width');

    app.TabTracking = uitab(app.TabGroup, 'Title', 'Tracking/Control');
    trackingGrid = uigridlayout(app.TabTracking, [2 2]);
    app.AxLatErr = uiaxes(trackingGrid); title(app.AxLatErr, 'Lateral error');
    app.AxYaw = uiaxes(trackingGrid); title(app.AxYaw, 'Yaw');
    app.AxSteer = uiaxes(trackingGrid); title(app.AxSteer, 'Steering');
    app.AxServo = uiaxes(trackingGrid); title(app.AxServo, 'Servo');

    app.TabTiming = uitab(app.TabGroup, 'Title', 'Timing');
    timingGrid = uigridlayout(app.TabTiming, [1 2]);
    app.AxCycle = uiaxes(timingGrid); title(app.AxCycle, 'Cycle time vs time');
    app.AxHist = uiaxes(timingGrid); title(app.AxHist, 'Cycle time histogram');

    infoPanel = uipanel(bottomGrid, 'Title', 'Thông tin');
    infoGrid = uigridlayout(infoPanel, [2 1]);
    app.InfoArea = uitextarea(infoGrid, 'Editable', 'off', ...
        'Value', {'Chọn 1 file CSV rồi bấm "Phân tích".'});
    app.KeyStatsArea = uitextarea(infoGrid, 'Editable', 'off', ...
        'Value', {'Các chỉ số chính sẽ hiện ở đây.'});

    %% callbacks
    function onBrowse(~, ~)
        [f, p] = uigetfile({'*.csv', 'CSV files (*.csv)'}, 'Chọn file CSV');
        if isequal(f, 0)
            return;
        end
        app.CurrentFile = fullfile(p, f);
        app.FileEdit.Value = app.CurrentFile;
        app.AnalyzeButton.Enable = 'on';
        app.ChangeOnlyButton.Enable = 'on';   % thêm dòng này
        app.StatusLabel.Text = 'Đã chọn file.';
    end

    function onBrowseOutput(~, ~)
        p = uigetdir(app.OutputEdit.Value, 'Chọn thư mục output');
        if isequal(p, 0)
            return;
        end
        app.OutputEdit.Value = p;
    end

    function onAnalyze(~, ~)
        if isempty(app.CurrentFile) || ~isfile(app.CurrentFile)
            uialert(app.UIFigure, 'Vui lòng chọn file CSV hợp lệ.', 'Thiếu file');
            return;
        end
        try
            app.StatusLabel.Text = 'Đang phân tích...';
            drawnow;

            T = readLogTable(app.CurrentFile);
            T = sanitizeTable(T);
            [~, fileStem, ~] = fileparts(app.CurrentFile);
            summary = summarizeFile(T, fileStem);
            maneuvers = analyzeManeuvers(T);

            result = struct();
            result.file = app.CurrentFile;
            result.table = T;
            result.summary = summary;
            result.maneuvers = maneuvers;
            app.LastResult = result;
            app.LastTable = T;

            app.SummaryTable.Data = summary.overall_table;
            app.ManeuverTable.Data = maneuvers.table;
            updateInfoBoxes(summary.overall_table, maneuvers.table, T);
            plotAll(T);

            if app.VisibleCheck.Value
                figs = makeStandalonePlots(T, fileStem, true);
                if app.SaveFigCheck.Value
                    outDir = fullfile(app.OutputEdit.Value, [fileStem '_analysis']);
                    if ~exist(outDir, 'dir'); mkdir(outDir); end
                    saveFigures(figs, outDir, fileStem);
                end
            end

            app.ExportButton.Enable = 'on';
            app.StatusLabel.Text = 'Phân tích xong.';
        catch ME
            app.StatusLabel.Text = 'Có lỗi.';
            uialert(app.UIFigure, getReport(ME, 'extended', 'hyperlinks', 'off'), 'Lỗi khi phân tích');
        end
    end
% Hàm chỉ phân tích CHANGE state 
    function onAnalyzeChangeOnly(~, ~)
    if isempty(app.CurrentFile) || ~isfile(app.CurrentFile)
        uialert(app.UIFigure, 'Vui lòng chọn file CSV hợp lệ.', 'Thiếu file');
        return;
    end
    try
        app.StatusLabel.Text = 'Đang phân tích CHANGE...';
        drawnow;

        T = readLogTable(app.CurrentFile);
        T = sanitizeTable(T);

        if ~hasVar(T, 'state')
            uialert(app.UIFigure, 'File không có cột state.', 'Thiếu dữ liệu');
            return;
        end

        % Dữ liệu CHANGE-only cho summary
        T_change = T(T.state == "CHANGE", :);

        if isempty(T_change)
            uialert(app.UIFigure, 'Không có mẫu nào ở trạng thái CHANGE.', 'Không có dữ liệu');
            return;
        end

        % Dữ liệu chỉ dùng để vẽ biểu đồ: từ CHANGE đầu tiên đến hết
        changeIdx = find(T.state == "CHANGE", 1, 'first');
        T_plot = T(changeIdx:end, :);

        [~, fileStem, ~] = fileparts(app.CurrentFile);
        changeStem = [fileStem '_CHANGE_ONLY'];

        summary = summarizeFile(T_change, changeStem);
        maneuvers = analyzeManeuvers(T);   % vẫn giữ nguyên như code cũ

        result = struct();
        result.file = app.CurrentFile;
        result.table = T_change;   % giữ nguyên dữ liệu phân tích
        result.summary = summary;
        result.maneuvers = maneuvers;
        app.LastResult = result;
        app.LastTable = T_change;

        app.SummaryTable.Data = summary.overall_table;
        app.ManeuverTable.Data = maneuvers.table;

        % Giữ nguyên info box theo CHANGE-only
        updateInfoBoxes(summary.overall_table, maneuvers.table, T_change);

        % Chỉ biểu đồ dùng dữ liệu từ CHANGE đầu tiên đến cuối file
        plotAll(T_plot);

        if app.VisibleCheck.Value
            figs = makeStandalonePlots(T_plot, changeStem, true);
            if app.SaveFigCheck.Value
                outDir = fullfile(app.OutputEdit.Value, [changeStem '_analysis']);
                if ~exist(outDir, 'dir'); mkdir(outDir); end
                saveFigures(figs, outDir, changeStem);
            end
        end
    
        app.ExportButton.Enable = 'on';
        app.StatusLabel.Text = 'Phân tích CHANGE xong.';
    catch ME
        app.StatusLabel.Text = 'Có lỗi.';
        uialert(app.UIFigure, getReport(ME, 'extended', 'hyperlinks', 'off'), 'Lỗi khi phân tích CHANGE');
    end
end

    function onExport(~, ~)
        if isempty(app.LastResult)
            uialert(app.UIFigure, 'Chưa có kết quả để xuất.', 'Thiếu dữ liệu');
            return;
        end
        try
            [~, fileStem, ~] = fileparts(app.CurrentFile);
            outDir = fullfile(app.OutputEdit.Value, [fileStem '_analysis']);
            if ~exist(outDir, 'dir'); mkdir(outDir); end

            if app.ExportCheck.Value
                writetable(app.LastResult.summary.overall_table, fullfile(outDir, [fileStem '_overall_summary.csv']));
                if ~isempty(app.LastResult.maneuvers.table)
                    writetable(app.LastResult.maneuvers.table, fullfile(outDir, [fileStem '_maneuver_summary.csv']));
                end
            end

            if app.SaveFigCheck.Value
                figs = makeStandalonePlots(app.LastTable, fileStem, false);
                saveNamedFigures(figs, outDir, fileStem);
                for ii = 1:numel(figs)
                    if isvalid(figs(ii))
                        close(figs(ii));
                    end
                end
            end

            app.StatusLabel.Text = 'Đã xuất kết quả.';
            uialert(app.UIFigure, ['Đã xuất kết quả vào: ' outDir], 'Hoàn tất', 'Icon', 'success');
        catch ME
            uialert(app.UIFigure, ME.message, 'Lỗi khi xuất');
        end
    end

    function onClear(~, ~)
        app.CurrentFile = '';
        app.LastResult = [];
        app.LastTable = table();
        app.FileEdit.Value = '';
        app.AnalyzeButton.Enable = 'off';
        app.ExportButton.Enable = 'off';
        app.ChangeOnlyButton.Enable = 'off'; % Thêm dòng này 
        app.SummaryTable.Data = table();
        app.ManeuverTable.Data = table();
        app.InfoArea.Value = {'Chọn 1 file CSV rồi bấm "Phân tích".'};
        app.KeyStatsArea.Value = {'Các chỉ số chính sẽ hiện ở đây.'};
        clearAllAxes();
        app.StatusLabel.Text = 'Đã xóa.';
    end

    %% helper UI functions
    function clearAllAxes()
        axList = [app.AxState, app.AxObs, app.AxMinDist, app.AxTTC, app.AxCost, app.AxLane, ...
                  app.AxLatErr, app.AxYaw, app.AxSteer, app.AxServo, app.AxCycle, app.AxHist];
        for iax = 1:numel(axList)
            cla(axList(iax));
            grid(axList(iax), 'on');
        end
    end

    function updateInfoBoxes(overall, maneuverTbl, T)
        info = {
            ['File: ' char(overall.file)];
            ['n_samples: ' num2str(overall.n_samples)];
            ['duration_s: ' fmt(overall.duration_s)];
            ['mean_cycle_ms: ' fmt(overall.mean_cycle_ms)];
            ['n_change_samples: ' num2str(overall.n_change_samples)];
            ['n_keep_samples: ' num2str(overall.n_keep_samples)];
            ['n_follow_samples: ' num2str(overall.n_follow_samples)]
        };

        key = {
            ['min_obstacle_distance_m: ' fmt(overall.min_obstacle_distance_m)];
            ['min_planner_distance_m: ' fmt(overall.min_planner_distance_m)];
            ['min_ttc_s: ' fmt(overall.min_ttc_s)];
            ['max_abs_lateral_error_m: ' fmt(overall.max_abs_lateral_error_m)];
            ['mean_abs_lateral_error_m: ' fmt(overall.mean_abs_lateral_error_m)];
            ['max_abs_yaw_rad: ' fmt(overall.max_abs_yaw_rad)];
            ['max_abs_steering_deg: ' fmt(overall.max_abs_steering_deg)];
            ['steering_sat_pct: ' fmt(overall.steering_sat_pct) ' %'];
            ['maneuver_count: ' num2str(height(maneuverTbl))]
        };

        if hasVar(T, 'cost')
            key{end+1} = ['mean_cost: ' fmt(mean(T.cost, 'omitnan'))];
            key{end+1} = ['max_cost: ' fmt(max(T.cost, [], 'omitnan'))];
        end

        app.InfoArea.Value = info;
        app.KeyStatsArea.Value = key;
    end

    function plotAll(T)
        clearAllAxes();
        if hasVar(T, 'time_ms') && height(T) > 0
            t = T.time_ms / 1000;
        else
            t = (0:height(T)-1)';
        end

        plotStateTimelineUI(app.AxState, t, T);
        plotMetricUI(app.AxObs, t, T, 'obs_distance_m', 'Obstacle distance (m)');
        plotMetricUI(app.AxMinDist, t, T, 'min_distance_m', 'Min planner distance (m)');
        plotMetricUI(app.AxTTC, t, T, 'min_ttc_s', 'Min TTC (s)');
        plotMetricUI(app.AxCost, t, T, 'cost', 'Cost');
        plotMetricUI(app.AxLane, t, T, 'lane_width_px', 'Lane width (px)');

        plotMetricUI(app.AxLatErr, t, T, 'lateral_error_m', 'Lateral error (m)');
        plotMetricUI(app.AxYaw, t, T, 'yaw_rad', 'Yaw (rad)');
        plotMetricUI(app.AxSteer, t, T, 'steering_deg', 'Steering (deg)');
        plotMetricUI(app.AxServo, t, T, 'servo', 'Servo');

        if hasVar(T, 'time_ms') && height(T) > 1
            dt_ms = diff(T.time_ms);
            plot(app.AxCycle, t(2:end), dt_ms, '-o', 'MarkerSize', 3, 'LineWidth', 1.1);
            title(app.AxCycle, 'Cycle time vs time');
            xlabel(app.AxCycle, 'time (s)'); ylabel(app.AxCycle, 'cycle time (ms)'); grid(app.AxCycle, 'on');

            histogram(app.AxHist, dt_ms);
            title(app.AxHist, 'Cycle time histogram');
            xlabel(app.AxHist, 'cycle time (ms)'); ylabel(app.AxHist, 'count'); grid(app.AxHist, 'on');
        else
            text(app.AxCycle, 0.5, 0.5, 'Không có đủ time_ms', 'HorizontalAlignment', 'center');
            text(app.AxHist, 0.5, 0.5, 'Không có đủ time_ms', 'HorizontalAlignment', 'center');
        end
    end
end

%% ===== Analysis functions =====
function T = readLogTable(filePath)
T = readtable(filePath, 'Delimiter', ',', 'TextType', 'string');
if hasVar(T, 'state')
    T.state = upper(strtrim(string(T.state)));
end
if hasVar(T, 'direction')
    T.direction = upper(strtrim(string(T.direction)));
end
end

function T = sanitizeTable(T)
numCols = {'obs_distance_m','min_distance_m','min_ttc_s','cost','time_ms', ...
           'lateral_error_m','yaw_rad','steering_deg','servo','lane_width_px','meter_per_pixel'};
for i = 1:numel(numCols)
    vn = numCols{i};
    if hasVar(T, vn)
        T.(vn) = double(T.(vn));
    end
end
if hasVar(T, 'obs_distance_m')
    x = T.obs_distance_m; x(x < 0) = NaN; T.obs_distance_m = x;
end
if hasVar(T, 'min_distance_m')
    x = T.min_distance_m; x(x >= 900) = NaN; T.min_distance_m = x;
end
if hasVar(T, 'min_ttc_s')
    x = T.min_ttc_s; x(x >= 900) = NaN; T.min_ttc_s = x;
end
end

function summary = summarizeFile(T, fileStem)
N = height(T);
summary = struct();
if hasVar(T, 'time_ms') && N > 1
    dt_ms = diff(T.time_ms);
else
    dt_ms = NaN;
end

states = strings(0, 1);
if hasVar(T, 'state')
    states = T.state;
end

overall = table();
overall.file = string(fileStem);
overall.n_samples = N;
overall.start_time_ms = getFirst(T, 'time_ms');
overall.end_time_ms = getLast(T, 'time_ms');
overall.duration_s = (overall.end_time_ms - overall.start_time_ms) / 1000;
overall.mean_cycle_ms = nanmeanLocal(dt_ms);
overall.max_cycle_ms = nanmaxLocal(dt_ms);
overall.min_cycle_ms = nanminLocal(dt_ms);
overall.n_change_samples = sum(states == "CHANGE");
overall.n_keep_samples = sum(states == "KEEP");
overall.n_follow_samples = sum(states == "FOLLOW");
overall.min_obstacle_distance_m = nanminOrNaN(T, 'obs_distance_m');
overall.min_planner_distance_m = nanminOrNaN(T, 'min_distance_m');
overall.min_ttc_s = nanminOrNaN(T, 'min_ttc_s');
overall.max_abs_lateral_error_m = nanmaxAbsOrNaN(T, 'lateral_error_m');
overall.mean_abs_lateral_error_m = nanmeanAbsOrNaN(T, 'lateral_error_m');
overall.max_abs_yaw_rad = nanmaxAbsOrNaN(T, 'yaw_rad');
overall.mean_abs_yaw_rad = nanmeanAbsOrNaN(T, 'yaw_rad');
overall.max_abs_steering_deg = nanmaxAbsOrNaN(T, 'steering_deg');
overall.mean_abs_steering_deg = nanmeanAbsOrNaN(T, 'steering_deg');
if hasVar(T, 'steering_deg')
    steer = T.steering_deg;
    overall.steering_sat_pct = 100 * mean(abs(steer) >= 24.999, 'omitnan');
else
    overall.steering_sat_pct = NaN;
end
if hasVar(T, 'lane_width_px')
    overall.mean_lane_width_px = mean(T.lane_width_px, 'omitnan');
    overall.std_lane_width_px = std(T.lane_width_px, 'omitnan');
else
    overall.mean_lane_width_px = NaN;
    overall.std_lane_width_px = NaN;
end
if hasVar(T, 'meter_per_pixel')
    overall.mean_meter_per_pixel = mean(T.meter_per_pixel, 'omitnan');
else
    overall.mean_meter_per_pixel = NaN;
end
summary.overall_table = overall;
summary.dt_ms = dt_ms;
end

function maneuvers = analyzeManeuvers(T)
maneuvers = struct();
maneuvers.table = table();
if ~hasVar(T, 'state') || ~hasVar(T, 'time_ms') || height(T) == 0
    return;
end
isChange = T.state == "CHANGE";
d = diff([false; isChange; false]);
starts = find(d == 1);
stops = find(d == -1) - 1;
if isempty(starts)
    return;
end
rows = table();
for i = 1:numel(starts)
    idx = starts(i):stops(i);
    seg = T(idx, :);
    row = table();
    row.maneuver_id = i;
    row.start_index = starts(i);
    row.end_index = stops(i);
    row.start_time_ms = seg.time_ms(1);
    row.end_time_ms = seg.time_ms(end);
    row.duration_s = (row.end_time_ms - row.start_time_ms) / 1000;
    if hasVar(seg, 'direction')
        dirs = seg.direction(seg.direction ~= "" & seg.direction ~= "NONE");
        if isempty(dirs)
            row.direction = "UNKNOWN";
        else
            row.direction = modeString(dirs);
        end
    else
        row.direction = "UNKNOWN";
    end
    row.min_obstacle_distance_m = nanminOrNaN(seg, 'obs_distance_m');
    row.min_planner_distance_m = nanminOrNaN(seg, 'min_distance_m');
    row.min_ttc_s = nanminOrNaN(seg, 'min_ttc_s');
    row.min_cost = nanminOrNaN(seg, 'cost');
    row.max_cost = nanmaxOrNaN(seg, 'cost');
    row.max_abs_lateral_error_m = nanmaxAbsOrNaN(seg, 'lateral_error_m');
    row.mean_abs_lateral_error_m = nanmeanAbsOrNaN(seg, 'lateral_error_m');
    row.max_abs_yaw_rad = nanmaxAbsOrNaN(seg, 'yaw_rad');
    row.max_abs_steering_deg = nanmaxAbsOrNaN(seg, 'steering_deg');
    if hasVar(seg, 'steering_deg')
        row.steering_sat_pct = 100 * mean(abs(seg.steering_deg) >= 24.999, 'omitnan');
    else
        row.steering_sat_pct = NaN;
    end
    if isempty(rows)
        rows = row;
    else
        rows = [rows; row]; %#ok<AGROW>
    end
end
maneuvers.table = rows;
end

%% ===== Plot functions =====
function plotStateTimelineUI(ax, t, T)
cla(ax);
if ~hasVar(T, 'state')
    text(ax, 0.5, 0.5, 'state column not found', 'HorizontalAlignment', 'center');
    axis(ax, 'off');
    return;
end
states = T.state;
y = nan(size(states));
y(states == "KEEP") = 0;
y(states == "CHANGE") = 1;
y(states == "FOLLOW") = 2;
plot(ax, t, y, 'LineWidth', 1.5);
yticks(ax, [0 1 2]);
yticklabels(ax, {'KEEP', 'CHANGE', 'FOLLOW'});
xlabel(ax, 'time (s)');
ylabel(ax, 'state');
grid(ax, 'on');
end

function plotMetricUI(ax, t, T, vn, yl)
cla(ax);
if ~hasVar(T, vn)
    text(ax, 0.5, 0.5, [vn ' not found'], 'HorizontalAlignment', 'center');
    axis(ax, 'off');
    return;
end
plot(ax, t, T.(vn), 'LineWidth', 1.2);
xlabel(ax, 'time (s)');
ylabel(ax, yl);
grid(ax, 'on');
end

function figs = makeStandalonePlots(T, fileStem, visible)
figs = gobjects(0);
vis = ternary(visible, 'on', 'off');
if hasVar(T, 'time_ms') && height(T) > 0
    t = T.time_ms / 1000;
else
    t = (0:height(T)-1)';
end
f1 = figure('Name', [fileStem ' - Planner Metrics'], 'Visible', vis, 'Color', 'w');
tiledlayout(f1, 3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
nexttile; plotStateStandalone(t, T); title('State timeline'); grid on;
nexttile; plotIfExists(t, T, 'obs_distance_m', 'Obstacle distance (m)'); title('Obstacle distance'); grid on;
nexttile; plotIfExists(t, T, 'min_distance_m', 'Min planner distance (m)'); title('Planner min distance'); grid on;
nexttile; plotIfExists(t, T, 'min_ttc_s', 'Min TTC (s)'); title('Planner TTC'); grid on;
nexttile; plotIfExists(t, T, 'cost', 'Cost'); title('Planner cost'); grid on;
nexttile; plotIfExists(t, T, 'lane_width_px', 'Lane width (px)'); title('Lane width'); grid on; xlabel('time (s)');
figs(end+1) = f1;

f2 = figure('Name', [fileStem ' - Tracking Metrics'], 'Visible', vis, 'Color', 'w');
tiledlayout(f2, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
nexttile; plotIfExists(t, T, 'lateral_error_m', 'Lateral error (m)'); title('Lateral error'); grid on;
nexttile; plotIfExists(t, T, 'yaw_rad', 'Yaw (rad)'); title('Yaw'); grid on;
nexttile; plotIfExists(t, T, 'steering_deg', 'Steering (deg)'); title('Steering'); grid on;
nexttile; plotIfExists(t, T, 'servo', 'Servo'); title('Servo'); grid on; xlabel('time (s)');
figs(end+1) = f2;

if hasVar(T, 'time_ms') && height(T) > 1
    dt_ms = diff(T.time_ms);
    f3 = figure('Name', [fileStem ' - Timing'], 'Visible', vis, 'Color', 'w');
    tiledlayout(f3, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile; plot(t(2:end), dt_ms, '-o', 'MarkerSize', 3); ylabel('cycle time (ms)'); xlabel('time (s)'); title('Cycle time vs time'); grid on;
    nexttile; histogram(dt_ms); xlabel('cycle time (ms)'); ylabel('count'); title('Cycle time histogram'); grid on;
    figs(end+1) = f3;
end
end


function plotStateStandalone(t, T)
if ~hasVar(T, 'state')
    text(0.5, 0.5, 'state column not found', 'HorizontalAlignment', 'center');
    axis off;
    return;
end
states = T.state;
y = nan(size(states));
y(states == "KEEP") = 0;
y(states == "CHANGE") = 1;
y(states == "FOLLOW") = 2;
plot(t, y, 'LineWidth', 1.5);
yticks([0 1 2]);
yticklabels({'KEEP','CHANGE','FOLLOW'});
xlabel('time (s)');
ylabel('state');
end

function plotIfExists(t, T, vn, yl)
if ~hasVar(T, vn)
    text(0.5, 0.5, sprintf('%s not found', vn), 'HorizontalAlignment', 'center');
    axis off;
    return;
end
plot(t, T.(vn), 'LineWidth', 1.2);
xlabel('time (s)');
ylabel(yl);
end

function saveFigures(figs, outDir, fileStem)
for i = 1:numel(figs)
    f = figs(i);
    pngPath = fullfile(outDir, sprintf('%s_fig%d.png', fileStem, i));
    exportgraphics(f, pngPath, 'Resolution', 180);
end
end

function saveNamedFigures(figs, outDir, fileStem)
nameList = {'planner_safety', 'tracking_control', 'timing'};
for i = 1:numel(figs)
    f = figs(i);
    if i <= numel(nameList)
        pngPath = fullfile(outDir, sprintf('%s_%s.png', fileStem, nameList{i}));
    else
        pngPath = fullfile(outDir, sprintf('%s_fig%d.png', fileStem, i));
    end
    exportgraphics(f, pngPath, 'Resolution', 180);
end
end

%% ===== Utilities =====
function tf = hasVar(T, vn)
tf = ismember(vn, T.Properties.VariableNames);
end

function v = getFirst(T, vn)
if hasVar(T, vn) && ~isempty(T.(vn))
    v = T.(vn)(1);
else
    v = NaN;
end
end

function v = getLast(T, vn)
if hasVar(T, vn) && ~isempty(T.(vn))
    v = T.(vn)(end);
else
    v = NaN;
end
end

function v = nanminOrNaN(T, vn)
if hasVar(T, vn)
    x = T.(vn);
    if isempty(x) || all(isnan(x))
        v = NaN;
    else
        v = min(x, [], 'omitnan');
    end
else
    v = NaN;
end
end

function v = nanmaxOrNaN(T, vn)
if hasVar(T, vn)
    x = T.(vn);
    if isempty(x) || all(isnan(x))
        v = NaN;
    else
        v = max(x, [], 'omitnan');
    end
else
    v = NaN;
end
end

function v = nanmaxAbsOrNaN(T, vn)
if hasVar(T, vn)
    x = abs(T.(vn));
    if isempty(x) || all(isnan(x))
        v = NaN;
    else
        v = max(x, [], 'omitnan');
    end
else
    v = NaN;
end
end

function v = nanmeanAbsOrNaN(T, vn)
if hasVar(T, vn)
    x = abs(T.(vn));
    if isempty(x) || all(isnan(x))
        v = NaN;
    else
        v = mean(x, 'omitnan');
    end
else
    v = NaN;
end
end

function v = nanmeanLocal(x)
if isempty(x) || all(isnan(x))
    v = NaN;
else
    v = mean(x, 'omitnan');
end
end

function v = nanminLocal(x)
if isempty(x) || all(isnan(x))
    v = NaN;
else
    v = min(x, [], 'omitnan');
end
end

function v = nanmaxLocal(x)
if isempty(x) || all(isnan(x))
    v = NaN;
else
    v = max(x, [], 'omitnan');
end
end

function s = modeString(strVec)
[u, ~, idx] = unique(strVec);
counts = accumarray(idx, 1);
[~, imax] = max(counts);
s = u(imax);
end

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end

function s = fmt(v)
if isempty(v) || (isscalar(v) && isnan(v))
    s = 'NaN';
else
    s = num2str(v, '%.6g');
end
end
