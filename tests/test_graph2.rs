#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;
use RayBNN_Graph;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;




#[test]
fn test_graph() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);





    let neuron_size: u64 = 16;

    let matrix_dims = arrayfire::Dim4::new(&[neuron_size,neuron_size,1,1]);
    let mut W: arrayfire::Array<f64>  = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire(
    	"./test_data/sparse_test3.csv",
    );

    W = arrayfire::sparse_from_dense(&W, arrayfire::SparseFormat::COO);


    let mut WValues = arrayfire::sparse_get_values(&W);
	let mut WRowIdxCOO = arrayfire::sparse_get_row_indices(&W);
	let mut WColIdx = arrayfire::sparse_get_col_indices(&W);



	let netdata: clusterdiffeq::neural::network_f64::network_metadata_type = clusterdiffeq::neural::network_f64::network_metadata_type {
		neuron_size: neuron_size,
	    input_size: 3,
		output_size: 2,
		proc_num: 3,
		active_size: neuron_size,
		space_dims: 3,
		step_num: 100,
        batch_size: 1,
		del_unused_neuron: true,

		time_step: 0.1,
		nratio: 0.5,
		neuron_std: 0.3,
		sphere_rad: 0.9,
		neuron_rad: 0.1,
		con_rad: 0.6,
        init_prob: 0.5,
        add_neuron_rate: 0.0,
		del_neuron_rate: 0.0,
		center_const: 0.005,
		spring_const: 0.01,
		repel_const: 10.0
	};

    let neuron_idx_dims = arrayfire::Dim4::new(&[1,13,1,1]);
    let mut neuron_idx: arrayfire::Array<i32> = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire(
    	"./test_data/neuron_idx3.csv",
    );

    neuron_idx = arrayfire::transpose(&neuron_idx, false);


    let neuron_pos_dims = arrayfire::Dim4::new(&[13,3,1,1]);
    let mut neuron_pos: arrayfire::Array<f64> = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire(
    	"./test_data/neuron_pos3.csv",
    );



    clusterdiffeq::graph::adjacency_f64::select_forward_sphere(
        &netdata, 
        &mut WValues, 
        &mut WRowIdxCOO, 
        &mut WColIdx, 
        &neuron_pos, 
        &neuron_idx
    );












    let mut WValues_cpu = vec!(f64::default();WValues.elements());
    WValues.host(&mut WValues_cpu);

    let mut WValues_act:Vec<f64> = vec![ -2.000000 
    ,-7.000000 
    ,-3.000000 
    ,-9.100000 
    ,-2.100000 
    ,-9.000000 
    ,-4.000000 
    ,-7.000000 
    ,-6.000000 
    ,-99.000000 
    ,-75.000000 
    ,-3.600000 
    ,-2.000000 
    ,-50.000000 
    ,-43.000000 
    ,-22.000000 
    ,-0.300000 
    ,-4.100000 
    ,-0.600000 ];

    WValues_cpu = WValues_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    WValues_act = WValues_act.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();


    assert_eq!(WValues_cpu, WValues_act);
















    let mut WRowIdxCOO_cpu = vec!(i32::default();WRowIdxCOO.elements());
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);


    let mut WRowIdxCOO_act:Vec<i32> = vec![
    4 ,
    4 ,
    5 ,
    6 ,
    3 ,
    5 ,
    6 ,
    9 ,
    10 ,
    11 ,
    12 ,
    9 ,
    10 ,
    11 ,
    12 ,
    14 ,
    15 ,
    14 ,
    15 ];


    assert_eq!(WRowIdxCOO_act, WRowIdxCOO_cpu);



















    let mut WColIdx_cpu = vec!(i32::default();WColIdx.elements());
    WColIdx.host(&mut WColIdx_cpu);



    let mut WColIdx_act:Vec<i32> = vec![
        0, 
        1 ,
        1 ,
        1 ,
        2 ,
        2 ,
        2 ,
        3 ,
        3 ,
        3 ,
        3 ,
        6 ,
        6 ,
        6 ,
        6 ,
       10 ,
       10 ,
       11 ,
       11 
     ];

    assert_eq!(WColIdx_cpu, WColIdx_act);







}
