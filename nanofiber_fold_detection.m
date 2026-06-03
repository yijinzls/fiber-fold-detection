function results = nanofiber_fold_detection(inputImagePath, varargin)
%NANOFIBER_FOLD_DETECTION Detect fold-like boundary features in microscopy images.
%
%   RESULTS = NANOFIBER_FOLD_DETECTION(INPUTIMAGEPATH) reads the image at
%   INPUTIMAGEPATH and detects fold-like points on connected edge boundaries.
%
%   Optional name-value parameters:
%       'MinBoundaryLength'          default 20
%       'CurvatureThreshold'         default 0.3
%       'AngleThreshold'             default pi/6
%       'MinDistanceBetweenFolds'    default 8
%       'ShowFigure'                 default true

    parser = inputParser;
    addRequired(parser, 'inputImagePath', @(x) ischar(x) || isstring(x));
    addParameter(parser, 'MinBoundaryLength', 20, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(parser, 'CurvatureThreshold', 0.3, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(parser, 'AngleThreshold', pi/6, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(parser, 'MinDistanceBetweenFolds', 8, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(parser, 'ShowFigure', true, @(x) islogical(x) && isscalar(x));
    parse(parser, inputImagePath, varargin{:});

    inputImagePath = char(parser.Results.inputImagePath);
    minBoundaryLength = parser.Results.MinBoundaryLength;
    curvatureThreshold = parser.Results.CurvatureThreshold;
    angleThreshold = parser.Results.AngleThreshold;
    minDistanceBetweenFolds = parser.Results.MinDistanceBetweenFolds;
    showFigure = parser.Results.ShowFigure;

    img = imread(inputImagePath);
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end

    grayImg = imadjust(grayImg);
    grayImg = imgaussfilt(grayImg, 0.5);
    edgeMask = edge(grayImg, 'Canny');

    connectedComponents = bwconncomp(edgeMask);
    labeled = labelmatrix(connectedComponents);
    labelImage = label2rgb(labeled, 'jet', 'k', 'shuffle');

    numRegions = connectedComponents.NumObjects;
    foldCount = 0;
    regionFoldCounts = zeros(numRegions, 1);
    drawnBoundaries = cell(numRegions, 1);
    figureHandle = [];

    if showFigure
        figureHandle = figure('Position', [100, 100, 1200, 600]);
        subplot(1, 2, 1);
        imshow(labelImage);
        title('Connected regions');

        subplot(1, 2, 2);
        imshow(img);
        hold on;
    end

    fprintf('Processing %d connected regions...\n', numRegions);

    for i = 1:numRegions
        regionPixels = connectedComponents.PixelIdxList{i};
        [row, col] = ind2sub(size(edgeMask), regionPixels);

        boundary = bwtraceboundary(edgeMask, [row(1), col(1)], 'N', 8, Inf, 'clockwise');
        if isempty(boundary) || size(boundary, 1) < minBoundaryLength
            continue;
        end

        x = boundary(:, 2);
        y = boundary(:, 1);

        if isDuplicateBoundary(x, y, drawnBoundaries, i)
            continue;
        end

        drawnBoundaries{i} = [x, y];

        if numel(x) > 5
            windowSize = min(5, floor(numel(x) / 10));
            if windowSize >= 3
                x = movmean(x, windowSize);
                y = movmean(y, windowSize);
            end
        end

        curvatureFolds = detectCurvatureFolds(x, y, curvatureThreshold);
        angleFolds = detectAngleFolds(x, y, angleThreshold);
        filteredFolds = removeAdjacentFolds(curvatureFolds | angleFolds, minDistanceBetweenFolds);

        if showFigure
            plot(x, y, 'g', 'LineWidth', 1.5);

            curvatureFiltered = curvatureFolds & ~angleFolds & filteredFolds;
            angleFiltered = angleFolds & ~curvatureFolds & filteredFolds;
            mixedFiltered = curvatureFolds & angleFolds & filteredFolds;

            if any(curvatureFiltered)
                scatter(x(curvatureFiltered), y(curvatureFiltered), 15, 'r', 'filled', 'MarkerEdgeColor', 'k');
            end
            if any(angleFiltered)
                scatter(x(angleFiltered), y(angleFiltered), 12, 'b', 'filled', 'MarkerEdgeColor', 'k');
            end
            if any(mixedFiltered)
                scatter(x(mixedFiltered), y(mixedFiltered), 18, 'm', 'filled', 'MarkerEdgeColor', 'k');
            end
        end

        currentFolds = sum(filteredFolds);
        regionFoldCounts(i) = currentFolds;
        foldCount = foldCount + currentFolds;
    end

    if showFigure
        title(sprintf('Total folds: %d (red: curvature, blue: angle, magenta: mixed)', foldCount));
        legend('Boundary', 'Curvature fold', 'Angle fold', 'Mixed fold', 'Location', 'northeast');
        hold off;
    end

    [~, inputName, inputExt] = fileparts(inputImagePath);
    averageFoldsPerRegion = foldCount / max(1, numRegions);

    fprintf('\n=== Fold detection results ===\n');
    fprintf('Input image: %s\n', [inputName inputExt]);
    fprintf('Processed connected regions: %d\n', numRegions);
    fprintf('Detected total folds: %d\n', foldCount);
    fprintf('Average folds per region: %.2f\n', averageFoldsPerRegion);

    results = struct();
    results.inputImageFile = [inputName inputExt];
    results.numRegions = numRegions;
    results.totalFoldCount = foldCount;
    results.averageFoldsPerRegion = averageFoldsPerRegion;
    results.regionFoldCounts = regionFoldCounts;
    results.figureHandle = figureHandle;
end

function isDuplicate = isDuplicateBoundary(x, y, drawnBoundaries, currentIndex)
    isDuplicate = false;
    for j = 1:currentIndex-1
        if isempty(drawnBoundaries{j})
            continue;
        end

        previousX = drawnBoundaries{j}(:, 1);
        previousY = drawnBoundaries{j}(:, 2);
        if numel(x) == numel(previousX)
            distance = mean(sqrt((x - previousX).^2 + (y - previousY).^2));
            if distance < 5
                isDuplicate = true;
                return;
            end
        end
    end
end

function foldPoints = detectCurvatureFolds(x, y, threshold)
    n = length(x);
    foldPoints = false(n, 1);

    if n < 5
        return;
    end

    for i = 3:n-2
        xSegment = x(i-2:i+2);
        ySegment = y(i-2:i+2);

        dx = gradient(xSegment);
        dy = gradient(ySegment);
        ddx = gradient(dx);
        ddy = gradient(dy);

        centerIndex = 3;
        dxCenter = dx(centerIndex);
        dyCenter = dy(centerIndex);
        ddxCenter = ddx(centerIndex);
        ddyCenter = ddy(centerIndex);

        denominator = (dxCenter^2 + dyCenter^2)^(3/2);
        if denominator > 1e-6
            kappa = abs(dxCenter * ddyCenter - dyCenter * ddxCenter) / denominator;
            if kappa > threshold
                foldPoints(i) = true;
            end
        end
    end
end

function foldPoints = detectAngleFolds(x, y, angleThreshold)
    n = length(x);
    foldPoints = false(n, 1);

    if n < 3
        return;
    end

    for i = 2:n-1
        v1 = [x(i) - x(i-1), y(i) - y(i-1)];
        v2 = [x(i+1) - x(i), y(i+1) - y(i)];

        len1 = norm(v1);
        len2 = norm(v2);

        if len1 > 1e-6 && len2 > 1e-6
            cosAngle = dot(v1, v2) / (len1 * len2);
            cosAngle = max(-1, min(1, cosAngle));
            angle = acos(cosAngle);

            if angle > angleThreshold
                foldPoints(i) = true;
            end
        end
    end
end

function filteredPoints = removeAdjacentFolds(foldPoints, minDistance)
    foldIndices = find(foldPoints);
    if length(foldIndices) <= 1
        filteredPoints = foldPoints;
        return;
    end

    filteredIndices = [];
    i = 1;
    while i <= length(foldIndices)
        currentIndex = foldIndices(i);
        filteredIndices = [filteredIndices, currentIndex]; %#ok<AGROW>

        j = i + 1;
        while j <= length(foldIndices) && (foldIndices(j) - currentIndex) < minDistance
            j = j + 1;
        end

        i = j;
    end

    filteredPoints = false(size(foldPoints));
    filteredPoints(filteredIndices) = true;
end
