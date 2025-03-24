import React, { useState } from 'react';
import Modal from 'react-modal';
import image1 from '../assets/images/my.png';
import image2 from '../assets/images/ng.png';
import image3 from '../assets/images/in.png';

const images = [image1, image2, image3];

Modal.setAppElement('#root');

function ImageGallery() {
  const [modalIsOpen, setModalIsOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);

  const openModal = (image) => {
    setSelectedImage(image);
    setModalIsOpen(true);
  };

  const closeModal = () => {
    setModalIsOpen(false);
    setSelectedImage(null);
  };

  return (
    <div>
      <div className="image-gallery">
        {images.map((url, index) => (
          <img key={index} src={url} alt={`image-${index}`} onClick={() => openModal(url)} />
        ))}
      </div>

      <Modal isOpen={modalIsOpen} onRequestClose={closeModal} className="modal" overlayClassName="modal-overlay">
        <button onClick={closeModal}>Close</button>
        <div className="modal-content">
          {selectedImage && <img src={selectedImage} alt="Selected" className="selected-image" />}
          <div className="related-images">
            {images.map((url, index) => (
              <img key={index} src={url} alt={`related-image-${index}`} />
            ))}
          </div>
        </div>
      </Modal>
    </div>
  );
}

export default ImageGallery;
