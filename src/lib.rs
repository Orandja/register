use std::sync::PoisonError;
use std::sync::{RwLock, RwLockReadGuard};

/// An asynchronous, in memory, software level register.
/// Register, Deregister operations are single thread limited while 
/// multiple thread can read the register with the method hold()
pub struct Register<T> {
	bank: RwLock<Box<[Option<T>]>>,
	next_index: RwLock<usize>,
	backing_list: RwLock<Vec<usize>>,
	capacity: usize,
}

impl<T> Register<T> {

	/// Create a new register with `capacity` possible elements
	pub fn new(capacity: usize) -> Self {
		let mut vec = Vec::<Option<T>>::with_capacity(capacity);
		unsafe { vec.set_len(capacity) } // prevent capacity drop
		Register {
			bank: RwLock::new(vec.into_boxed_slice()),
			next_index: RwLock::default(),
			backing_list: RwLock::default(),
			capacity: capacity,
		}
	}

	/// Capacity of register
	#[inline]
	pub fn capacity(&self) -> usize {
		self.capacity
	}

	/// Remaining available space in register
	#[inline]
	pub fn available(&self) -> Result<usize, RegisterError> {
		Ok(self.capacity - *self.next_index.read()? + self.backing_list.read()?.len())
	}

	/// Find the next free available index
	#[inline]
	fn available_index(&self) -> Result<usize, RegisterError> {
		let mut backing_list = self.backing_list.write()?;
		Ok(match backing_list.pop() {
			Some(idx) => idx,
			None => {
				let mut idx = self.next_index.write()?;
				if self.capacity <= *idx {
					return Err(RegisterError::CapacityOverflow(self.capacity));
				}
				let result = idx.clone();
				*idx += 1;
				result
			}
		})
	}

	/// Register an item. return it's index
	pub fn register(&self, item: T) -> Result<usize, RegisterError> {
		let mut bank = self.bank.write()?; // must hold bank before modifying indexes
		let index = self.available_index()?;
		bank[index] = Some(item);
		Ok(index)
	}

	/// Deregister an item by it's index. return the deregistered value.
	pub fn deregister(&self, index: &usize) -> Result<Option<T>, RegisterError> {
		if self.capacity <= *index {
			return Err(RegisterError::InvalidIndex(self.capacity, *index));
		}
		if *index >= *self.next_index.read()? {
			return Ok(None);
		}
		let mut item: Option<T> = None;
		let mut bank = self.bank.write()?;
		unsafe { std::ptr::swap(&mut item, &mut bank[*index]) }
		self.backing_list.write()?.push(*index); // must be before drop(bank)
		drop(bank); // drop after indexes are order correctly
		Ok(item)
	}

	/// Read an index.
	/// Return a hold that access the underlying register's bank.
	/// Until ItemHold is drop the register is Read Lock.
	pub fn hold(&self, index: &usize) -> Result<ItemHold<'_, T>, RegisterError> {
		if self.capacity <= *index {
			return Err(RegisterError::InvalidIndex(self.capacity, *index));
		}

		// An index that is not yet register can't possibly exist in bank.
		// calling at index > next_index will panic since None isn't created.
		let idx = if *index < *self.next_index.read()? {
			Some(*index)
		} else {
			None
		};

		Ok(ItemHold {
			lock: self.bank.read()?,
			index: idx,
		})
	}

	/// Same as hold() without verifing anything.
	/// Used inside the iterator
	#[inline]
	fn unsafe_hold(&self, index: &usize) -> Result<ItemHold<'_, T>, RegisterError> {
		Ok(ItemHold {
			lock: self.bank.read()?,
			index: Some(*index),
		})
	}
}

/// because compiler say Register is not Send friendly
unsafe impl<T> Send for Register<T> {}
unsafe impl<T> Sync for Register<T> {}

/// A Read lock hold of an item inside the register
#[derive(Debug)]
pub struct ItemHold<'a, T> {
	lock: RwLockReadGuard<'a, Box<[Option<T>]>>,
	index: Option<usize>,
}

impl<'a, T> std::ops::Deref for ItemHold<'a, T> {
	type Target = Option<T>;

	fn deref(&self) -> &Self::Target {
		match self.index {
			Some(idx) => &self.lock[idx],
			None => &None,
		}
	}
}

/// Comparing something like `Option<ItemHold<'_, T: PartialEq + Eq>`
/// isn't possible without this.
impl<'a, T: PartialEq + Eq> PartialEq for ItemHold<'a, T> {
	fn eq(&self, other: &Self) -> bool {
		// Compare the actual value not the `hold` struct
		**self == **other
	}
}

/// An iterator over the values inside the register
pub struct RegisterIterator<'a, T> {
	inner: &'a Register<T>,
	index: usize,
}

impl<'a, T> Iterator for RegisterIterator<'a, T> {
	type Item = ItemHold<'a, T>;
	fn next(&mut self) -> Option<Self::Item> {
		loop {
			let max = match self.inner.next_index.read() {
				Err(_) => return None,
				Ok(idx) => *idx,
			};
			if self.index >= max { 
				return None;
			}
			drop(max);
			match self.inner.unsafe_hold(&self.index) {
				Err(_) => return None,
				Ok(item) => {
					self.index += 1;
					return Some(item)
				}
			}
		}
	}
}

impl<'a, T> IntoIterator for &'a Register<T> {
	type Item = ItemHold<'a, T>;
	type IntoIter = RegisterIterator<'a, T>;

	fn into_iter(self) -> Self::IntoIter {
		RegisterIterator {
			inner: self,
			index: 0,
		}
	}
}

/// Produced error by Register
#[derive(Debug)]
pub enum RegisterError {
	
	/// the specified index can't used.
	/// happen when deregister() or hold()
	InvalidIndex(usize, usize),

	/// The register is full.
	CapacityOverflow(usize),
	
	/// A thread panicked during an operation.
	/// Thus register is poisoned
	Poisoned,
}

impl std::error::Error for RegisterError {}
impl std::fmt::Display for RegisterError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			RegisterError::InvalidIndex(access, limit) => write!(
				f,
				"Accessing index {} but bank can only contains {} element(s)",
				access, limit
			),
			RegisterError::CapacityOverflow(limit) => write!(
				f,
				"`register()`'s requests exeed capacity. Capacity: {}",
				limit
			),
			RegisterError::Poisoned => write!(f, "poisoned register"),
		}
	}
}

// TODO: better poison handling. While Reading a poison can't harm the register.
impl<T> From<PoisonError<T>> for RegisterError {
	fn from(_: PoisonError<T>) -> Self {
		RegisterError::Poisoned
	}
}

#[cfg(test)]
pub mod tests {
	use crate::*;
	use std::sync::{
		atomic::{AtomicBool, Ordering},
		Arc, Mutex,
	};
	use std::thread;

	// test each functions to behave as expected
	#[test]
	fn single_thread() {
		let bank = Register::<String>::new(5);

		assert_eq!(5, bank.capacity());

		let _idx_1 = bank.register("1".into()).unwrap();
		let idx_2 = bank.register("2".into()).unwrap();
		let idx_3 = bank.register("3".into()).unwrap();
		let _idx_4 = bank.register("4".into()).unwrap();

		assert_eq!("2".to_string(), bank.deregister(&idx_2).unwrap().unwrap());

		assert_eq!(*bank.hold(&idx_2).unwrap(), None);
		assert_eq!(*bank.hold(&idx_3).unwrap(), Some("3".into()));


		//assert_eq!(2, bank.available().unwrap());

		let idx_2_bis = bank.register("6".into()).unwrap();
		assert_eq!(idx_2, idx_2_bis);

		assert_eq!("3", bank.deregister(&idx_3).unwrap().unwrap());

		let iter = &mut bank.into_iter();
		assert_eq!(*iter.next().unwrap(), Some("1".into()));
		assert_eq!(*iter.next().unwrap(), Some("6".into()));
		assert_eq!(*iter.next().unwrap(), None);

		bank.register("7".into()).unwrap();
		bank.register("8".into()).unwrap();

		assert_eq!(*iter.next().unwrap(), Some("4".into()));
		assert_eq!(*iter.next().unwrap(), Some("8".into()));
		assert_eq!(iter.next(), None);

	}

	// FULL MESS! do not judge please :)
	// just to be sure hold() / register() / deregister()
	// do not collide with each other
	#[test]
	fn multi_thread() {
		let share = Arc::new(Register::<String>::new(1000));
		// Write into `actions` what threads are doing.
		let actions = Arc::new(Mutex::new(Vec::<String>::new()));
		let mut joins = vec![];
		// Just a bunch of thread that register / deregister items.
		for i in 0..10 {
			let writer_share = share.clone();
			let actions_share = actions.clone();
			joins.push(thread::spawn(move || {
				let mut indexes = vec![];
				for j in 0..100 {
					indexes.push(
						writer_share
							.register(format!("Thread {}, count : {}", i, j))
							.unwrap(),
					);
					actions_share.lock().unwrap().push("register 1".into());
				}
				for u in 0usize..50 {
					writer_share.deregister(&indexes[u * 2]).unwrap();

					actions_share.lock().unwrap().push("deregister".into());

					indexes[u * 2] = writer_share
						.register(format!("Thread {}, replaced : {}", i, u))
						.unwrap();

					actions_share.lock().unwrap().push("register 2".into());
				}
			}));
		}

		// a thread that constantly read the register while upper thread modify it.
		let reader_share = share.clone();
		let actions_share = actions.clone();
		let stop = Arc::new(AtomicBool::new(false));
		let stop_share = stop.clone();
		thread::spawn(move || 'out: loop {
			for _ in &*reader_share {
				actions_share.lock().unwrap().push(format!("read"));
				if stop_share.load(Ordering::Relaxed) {
					break 'out;
				}
			}
		});

		for join in joins {
			join.join().unwrap();
		}
		stop.store(true, Ordering::Relaxed);

		// visual verification of all threads reading / register / deregister simultaneously
		// println!("{:#?}", actions.lock().unwrap());

		// if everything goes well, register is full
		assert_eq!(share.available().unwrap(), 0)
	}
}
