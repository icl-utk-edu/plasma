
--------------------------------------------------------------------------------
function gbtrf_nb (type, num_threads, n, bw)
        return 256
end

function gbtrf_max_panel_threads (type, num_threads, n, bw)
        return 1
end

--------------------------------------------------------------------------------
function geadd_nb (type, num_threads, m, n)
        return 256
end

--------------------------------------------------------------------------------
function geinv_nb (type, num_threads, m, n)
        return 256
end

function geinv_ib (type, num_threads, m, n)
	return 64
end

function geinv_max_panel_threads (type, num_threads, m, n)
	return 1
end
--------------------------------------------------------------------------------
function gelqf_nb (type, num_threads, m, n)
        return 256
end

function gelqf_ib (type, num_threads, m, n)
        return 64
end

--------------------------------------------------------------------------------
function gemm_nb (type, num_threads, m, n, k)
        return 256
end

--------------------------------------------------------------------------------
function geqrf_nb (type, num_threads, m, n)
        return 256
end

function geqrf_ib (type, num_threads, m, n)
        return 64
end

--------------------------------------------------------------------------------
function geswp_nb (type, num_threads, m, n)
        return 256
end

--------------------------------------------------------------------------------
function getrf_nb (type, num_threads, m, n)
	return 256
end

function getrf_ib (type, num_threads, m, n)
	return 64
end

function getrf_max_panel_threads (type, num_threads, m, n)
	return 1
end

--------------------------------------------------------------------------------
function hetrf_nb (type, num_threads, n)
        return 256
end

--------------------------------------------------------------------------------
function lacpy_nb (type, num_threads, m, n)
        return 256
end

--------------------------------------------------------------------------------
function lag2c_nb (type, num_threads, m, n)
        return 256
end

--------------------------------------------------------------------------------
function lange_nb (type, num_threads, m, n)
        return 256
end

--------------------------------------------------------------------------------
function lansy_nb (type, num_threads, n)
        return 256
end

--------------------------------------------------------------------------------
function lantr_nb (type, num_threads, m, n)
        return 256
end

--------------------------------------------------------------------------------
function lascl_nb (type, num_threads, m, n)
        return 256
end

--------------------------------------------------------------------------------
function laset_nb (type, num_threads, m, n)
        return 256
end

--------------------------------------------------------------------------------
function lauum_nb (type, num_threads, n)
        return 256
end

--------------------------------------------------------------------------------
function pbtrf_nb (type, num_threads, n)
        return 256
end

--------------------------------------------------------------------------------
function potrf_nb (type, num_threads, n)
        return 256
end

--------------------------------------------------------------------------------
function poinv_nb (type, num_threads, n)
        return 256
end

--------------------------------------------------------------------------------
function symm_nb (type, num_threads, m, n)
        return 256
end

--------------------------------------------------------------------------------
function syr2k_nb (type, num_threads, n, k)
        return 256
end

--------------------------------------------------------------------------------
function syrk_nb (type, num_threads, n, k)
        return 256
end

--------------------------------------------------------------------------------
function tradd_nb (type, num_threads, m, n)
        return 256
end

--------------------------------------------------------------------------------
function trmm_nb (type, num_threads, m, n)
        return 256
end

--------------------------------------------------------------------------------
function trsm_nb (type, num_threads, m, n)
        return 256
end
