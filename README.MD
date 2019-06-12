# Orandja register

An asyncronous, in memory, software level register.

# Usage

```rust
    //create a register of max 5 strings
    let reg = std::sync::Arc::new(Register::<String>::new(5));

    // Registering "Hello", "World". Give indexes
    let index_1 = reg.register("Hello".into()).unwrap(); // index_1 == 0
    let index_2 = reg.register("World".into()).unwrap(); // index_2 == 1
    
    // deregister "Hello"
    let _deregister: Option<String> = reg.deregister(&index_1).unwrap();

    // pass register to thread
    let reg_clone = reg.clone();
    thread::spawn(move || {
        // hold a read lock to "World"
        // until drop(hold). `reg` is unable to register/deregister
        let hold = reg_clone.hold(&index_2).unwrap();
        assert_eq!(Some("World".to_string()), *hold2);
        thread::sleep_ms(20);
        drop(hold); // unnesesary, just to demonstrate
    });

    thread::sleep_ms(10);
    // Wait until thread drop hold2
    // deregister index_2
    reg.deregister(&index_2).unwrap();
```