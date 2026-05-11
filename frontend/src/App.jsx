import { useState } from 'react';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fungsi untuk menangani saat pengguna memilih gambar
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null); // Reset hasil jika unggah gambar baru
      setError(null);
    }
  };

  // Fungsi untuk mengirim gambar ke Backend FastAPI
  const handlePredict = async () => {
    if (!image) return;
    
    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      // Menembak API Backend Lokal Anda
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Gagal terhubung ke server backend. Pastikan server FastAPI menyala.");
      }

      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Fungsi utilitas untuk memberikan warna teks berdasarkan hasil
  const getStatusColor = (status) => {
    if (status === "Normal") return "text-emerald-600";
    if (status === "Pneumonia") return "text-red-600";
    return "text-yellow-600"; // Untuk "Bukan Rontgen"
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      {/* Kartu Utama */}
      <div className="max-w-md w-full bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
        
        {/* Header Tema Soft Mint */}
        <div className="bg-soft-mint p-6 text-center shadow-sm">
          <h1 className="text-2xl font-bold text-teal-900">Deteksi Pneumonia</h1>
          <p className="text-sm mt-1 text-teal-800 font-medium">Unggah citra X-Ray dada untuk dianalisis oleh AI</p>
        </div>

        <div className="p-6">
          {/* Area Drag & Drop / Input File */}
          <div className="flex items-center justify-center w-full mb-5">
            <label className="flex flex-col items-center justify-center w-full h-40 border-2 border-soft-mint border-dashed rounded-xl cursor-pointer bg-gray-50 hover:bg-teal-50 transition-all">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg className="w-10 h-10 mb-3 text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path>
                </svg>
                <p className="mb-1 text-sm text-gray-600"><span className="font-semibold text-teal-600">Klik untuk unggah</span></p>
                <p className="text-xs text-gray-400">JPG atau PNG (Maks 5MB)</p>
              </div>
              <input type="file" className="hidden" accept="image/png, image/jpeg, image/jpg" onChange={handleImageChange} />
            </label>
          </div>

          {/* Area Pratinjau Gambar */}
          {preview && (
            <div className="mb-5 rounded-xl overflow-hidden border border-gray-200 shadow-sm bg-gray-900">
              <img src={preview} alt="Preview Rontgen" className="w-full h-48 object-contain" />
            </div>
          )}

          {/* Tombol Analisis */}
          <button 
            onClick={handlePredict} 
            disabled={!image || loading}
            className={`w-full py-3 px-4 rounded-xl font-bold text-teal-900 transition-all flex justify-center items-center gap-2 
              ${!image || loading 
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed' 
                : 'bg-soft-mint hover:bg-teal-300 shadow-md hover:shadow-lg active:scale-95'}`}
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-teal-800" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Menganalisis...
              </>
            ) : 'Analisis Gambar'}
          </button>

          {/* Area Pesan Error */}
          {error && (
            <div className="mt-4 p-4 bg-red-50 text-red-600 border border-red-100 rounded-xl text-sm text-center font-medium">
              ⚠️ {error}
            </div>
          )}

          {/* Area Hasil Prediksi */}
          {result && (
            <div className="mt-6 p-5 bg-teal-50 border border-soft-mint rounded-xl text-center shadow-inner animate-fade-in">
              <h3 className="text-xs text-teal-600 uppercase tracking-widest font-bold mb-2">Hasil Diagnosa AI</h3>
              <div className={`text-3xl font-extrabold mb-1 ${getStatusColor(result.status)}`}>
                {result.status}
              </div>
              <div className="text-gray-600 font-medium mt-2">
                Confidence: <span className="text-gray-900 font-bold">{result.confidence}%</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;