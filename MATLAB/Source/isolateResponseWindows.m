function impulseIndices = isolateResponseWindows(response, thresholdMultiplier, fs)
    % isolateResponseWindows: Isolates response windows between impulse hits.
    %
    % Inputs:
    %   - response: Signal data to analyze for impulse detection.
    %   - thresholdMultiplier: Multiplier for the standard deviation to set the impulse threshold.
    %   - fs: Minimum spacing in samples between consecutive impulses.
    %
    % Outputs:
    %   - impulseIndices: Indices of detected impulses (peaks).
    %
    % Example Usage:
    %   impulseIndices = isolateResponseWindows(response, 5, 100);

    % Define the threshold for detecting impulses based on standard deviation
    threshold = std(response) * thresholdMultiplier;

    % Find sharp peaks using prominence
    [~, allImpulseIndices] = findpeaks(abs(response), ...
                                       'MinPeakHeight', threshold, ...
                                       'MinPeakProminence', threshold / 2); % Adjust prominence to match sharpness

    % Remove closely spaced indices (indices closer than fs samples apart)
    impulseIndices = [];
    for i = 1:length(allImpulseIndices)
        if isempty(impulseIndices) || (allImpulseIndices(i) - impulseIndices(end)) >= fs
            impulseIndices = [impulseIndices, allImpulseIndices(i)];
        end
    end

    % Display the number of impulses detected
    disp(['Number of detected impulses: ', num2str(length(impulseIndices))]);

    % Debugging output: Display indices
    if ~isempty(impulseIndices)
        disp('Impulse Indices:');
        disp(impulseIndices);
    else
        disp('No impulses detected. Try adjusting the thresholdMultiplier.');
    end
end
