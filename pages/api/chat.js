export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { messages } = req.body;

  try {
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'openai/gpt-3.5-turbo', // หรือเลือกโมเดลอื่น
        messages: [
          {
            role: 'system',
            content: 'คุณเป็นผู้ช่วยที่ตอบคำถามจากข้อมูลในไฟล์ PDF เท่านั้น',
          },
          ...messages,
        ],
      }),
    });

    const data = await response.json();
    res.status(200).json(data.choices[0].message);
  } catch (error) {
    res.status(500).json({ error: 'Error calling OpenRouter API' });
  }
}
