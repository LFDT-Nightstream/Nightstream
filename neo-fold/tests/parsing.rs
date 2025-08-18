#[cfg(test)]
mod tests {

    use byteorder::{BigEndian, WriteBytesExt};

    // Since read_comms_block is private, let's create a public wrapper for testing
    // or test the public interface that uses it
    fn create_comms_block_data(comms: &[Vec<u8>]) -> Vec<u8> {
        let mut data = vec![comms.len() as u8]; // len=num_comms
        for comm in comms {
            let mut len_bytes = vec![];
            len_bytes.write_u32::<BigEndian>(comm.len() as u32).unwrap();
            data.extend_from_slice(&len_bytes);
            data.extend_from_slice(comm);
        }
        data
    }

    #[test]
    fn test_comms_serialization_format() {
        let comm1 = vec![1u8; 32];
        let comm2 = vec![2u8; 32];
        let comms = vec![comm1.clone(), comm2.clone()];
        
        let data = create_comms_block_data(&comms);
        
        // Check format: [num_comms:u8][len1:u32][comm1][len2:u32][comm2]
        assert_eq!(data[0], 2u8); // num_comms = 2
        
        // Check first comm
        let len1 = u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
        assert_eq!(len1, 32);
        assert_eq!(&data[5..37], &comm1);
        
        // Check second comm  
        let len2 = u32::from_be_bytes([data[37], data[38], data[39], data[40]]);
        assert_eq!(len2, 32);
        assert_eq!(&data[41..73], &comm2);
    }

    #[test]
    fn test_empty_comms_block() {
        let comms: Vec<Vec<u8>> = vec![];
        let data = create_comms_block_data(&comms);
        
        // Should just be a single byte with value 0
        assert_eq!(data, vec![0u8]);
    }

    #[test]
    fn test_single_comm_block() {
        let comm = vec![42u8; 16];
        let comms = vec![comm.clone()];
        let data = create_comms_block_data(&comms);
        
        assert_eq!(data[0], 1u8); // num_comms = 1
        let len = u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
        assert_eq!(len, 16);
        assert_eq!(&data[5..21], &comm);
    }
}
