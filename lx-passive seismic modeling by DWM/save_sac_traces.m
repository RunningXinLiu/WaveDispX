function save_sac_traces(SR, sacfolder, i, origin_lat, origin_lon, dt, traces)
    % Great-circle distance and azimuth from local Cartesian offsets.
    distance_m = hypot(SR.rx(i), SR.ry(i));
    azimuth_deg = mod(atan2d(SR.rx(i), SR.ry(i)), 360);
    [lat, lon] = reckon(origin_lat, origin_lon, ...
        km2deg(distance_m / 1000), azimuth_deg);

    % Save x component
    filename_x = fullfile(sacfolder, sprintf('rec_%d_x.SAC', i));
    sacfile = create_sac_file(filename_x, traces.x(:, i), lat, lon, SR, i, dt, 'X');
    writesac(sacfile);

    % Save y component
    filename_y = fullfile(sacfolder, sprintf('rec_%d_y.SAC', i));
    sacfile = create_sac_file(filename_y, traces.y(:, i), lat, lon, SR, i, dt, 'Y');
    writesac(sacfile);

    % Save z component
    filename_z = fullfile(sacfolder, sprintf('rec_%d_z.SAC', i));
    sacfile = create_sac_file(filename_z, traces.z(:, i), lat, lon, SR, i, dt, 'Z');
    writesac(sacfile);
end
