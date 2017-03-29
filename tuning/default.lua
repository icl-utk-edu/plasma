
function getrf_nb (type, num_threads, m, n)
	return 256
end

function getrf_ib (type, num_threads, m, n)
	return 64
end

function getrf_max_panel_threads (type, num_threads, m, n)
	return 1
end
